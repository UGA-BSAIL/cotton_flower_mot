import enum
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from pydantic.dataclasses import dataclass

from .assignment import construct_gt_sinkhorn_matrix
from .config import ModelConfig
from .heat_maps import make_heat_map
from .schemas import ModelInputs, ModelTargets
from .schemas import ObjectTrackingFeatures as Otf

_FEATURE_DESCRIPTION = {
    Otf.IMAGE_HEIGHT.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_WIDTH.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_FILENAME.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    # TODO (danielp): These should be compressed into a single value per frame.
    Otf.IMAGE_SOURCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_FORMAT.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.OBJECT_BBOX_X_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_X_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_CLASS_TEXT.value: tf.io.RaggedFeature(tf.dtypes.string),
    Otf.OBJECT_CLASS_LABEL.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.OBJECT_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_SEQUENCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_FRAME_NUM.value: tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing flower annotations.
"""

Feature = Dict[str, tf.Tensor]
"""
Feature dictionary that contains only normal tensors.
"""
RaggedFeature = Dict[str, tf.RaggedTensor]
"""
Feature dictionary that contains only ragged tensors.
"""
MaybeRagged = Union[tf.Tensor, tf.RaggedTensor]
"""
Possibly a normal tensor or a ragged one.
"""
MaybeRaggedFeature = Dict[str, MaybeRagged]
"""
Feature dictionary that may contain normal or ragged tensors.
"""

_RANDOM_SEED = 2021
"""
Seed to use for random number generation.
"""

_RAGGED_INPUTS = {
    ModelInputs.DETECTIONS.value,
    ModelInputs.TRACKLETS.value,
    ModelInputs.DETECTION_GEOMETRY.value,
    ModelInputs.TRACKLET_GEOMETRY.value,
    ModelInputs.DETECTIONS_OFFSETS.value,
}
"""
Input features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_INPUTS = {
    ModelInputs.SEQUENCE_ID.value,
    ModelInputs.DETECTIONS_FRAME.value,
    ModelInputs.DETECTIONS_HEATMAP.value,
}
"""
Input features that the model expects to be normal tensors.
"""
_RAGGED_TARGETS = set()
"""
Target features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_TARGETS = {
    ModelTargets.SINKHORN.value,
    ModelTargets.ASSIGNMENT.value,
}
"""
Target features that the model expects to be normal tensors.
"""


@enum.unique
class FeatureName(enum.Enum):
    """
    Standard key names for processed features.
    """

    FRAME_IMAGE = "frame_image"
    """
    Full frame image.
    """
    HEAT_MAP = "heat_map"
    """
    Corresponding detection heatmap.
    """
    DETECTIONS_OFFSETS = "detections_offsets"
    """
    Pixel offsets for the detections in the heatmap.
    """
    DETECTIONS = "detections"
    """
    Extracted detection crops.
    """
    GEOMETRY = "geometry"
    """
    Geometric features.
    """
    OBJECT_IDS = "object_ids"
    """
    Object IDs.
    """
    FRAME_NUM = "frame_num"
    """
    The frame number in the underlying video clip.
    """
    SEQUENCE_ID = "sequence_id"
    """
    The sequence ID of the clip.
    """


_NUM_THREADS = cpu_count()
"""
Number of threads to use for multi-threaded operations.
"""


@dataclass(frozen=True)
class DataAugmentationConfig:
    """
    Configuration to use for data augmentation.

    Attributes:
        max_bbox_jitter: Maximum amount of jitter to add to the bounding boxes,
            as a fraction of the frame size.
        max_brightness_delta: Maximum amount to adjust the brightness by.
        max_hue_delta: Maximum amount to adjust the hue by.

        min_contrast: Minimum contrast to use.
        max_contrast: Maximum contrast to use.

        min_saturation: Minimum saturation factor to use.
        max_saturation: Maximum saturation factor to use.
    """

    max_bbox_jitter: float = 0.0
    max_brightness_delta: float = 0.0
    max_hue_delta: float = 0.0

    min_contrast: Optional[float] = None
    max_contrast: Optional[float] = None

    min_saturation: Optional[float] = None
    max_saturation: Optional[float] = None


def _add_bbox_jitter(
    bbox_coords: tf.Tensor,
    *,
    max_fractional_change: float = 0.05,
    image_shape: tf.Tensor,
) -> tf.Tensor:
    """
    Adds uniform random jitter to bounding box coordinates.

    Args:
        bbox_coords: The bounding box coordinates. Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`, in pixels.
        max_fractional_change: The maximum absolute amount that a coordinate
            can change, as a fraction of the original coordinate value.
        image_shape: A 1D vector describing the shape of the corresponding
            image.

    Returns:
        The bounding box coordinates with jitter.

    """
    # Calculate maximum change in pixels.
    image_height_width = tf.cast(image_shape[:2], tf.float32)
    max_absolute_change = image_height_width * max_fractional_change
    # Expand it so it aligns with both the corner points.
    max_absolute_change = tf.tile(max_absolute_change, (2,))

    # Generate random jitters.
    jitter_magnitude = tf.random.uniform(
        tf.shape(bbox_coords),
        minval=-max_absolute_change,
        maxval=max_absolute_change,
    )

    # Apply the jitters.
    return bbox_coords + jitter_magnitude


def _augment_images(
    images: tf.Tensor, config: DataAugmentationConfig
) -> tf.Tensor:
    """
    Applies data augmentation to images.

    Args:
        images: The images to augment.
        config: Configuration for data augmentation.

    Returns:
        The augmented images.

    """
    # Convert to floats once so we're not doing many redundant conversions.
    images = tf.cast(images, tf.float32)

    images = tf.image.random_brightness(images, config.max_brightness_delta)
    images = tf.image.random_hue(images, config.max_hue_delta)

    if config.min_contrast is not None and config.max_contrast is not None:
        images = tf.image.random_contrast(
            images, config.min_contrast, config.max_contrast
        )
    if config.min_saturation is not None and config.max_saturation is not None:
        images = tf.image.random_saturation(
            images, config.min_saturation, config.max_saturation
        )

    images = tf.clip_by_value(images, 0.0, 255.0)
    return tf.cast(images, tf.uint8)


def _get_geometric_features(
    bbox_coords: tf.Tensor, *, image_shape: tf.Tensor
) -> tf.Tensor:
    """
    Converts a batch of bounding boxes from the format in TFRecords to the
    single-tensor geometric feature format.

    Args:
        bbox_coords: The bounding box coordinates. Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`, in pixels.
        image_shape: A 1D vector describing the shape of the corresponding
            image.

    Returns:
        A tensor of shape [N, 4], where N is the total number of bounding
        boxes in the input. The ith row specifies the normalized coordinates of
        the ith bounding box in the form [center_x, center_y, width, height].

    """
    bbox_coords = tf.ensure_shape(bbox_coords, (None, 4))

    def _extract_features() -> tf.Tensor:
        x_min = bbox_coords[:, 1]
        x_max = bbox_coords[:, 3]
        y_min = bbox_coords[:, 0]
        y_max = bbox_coords[:, 2]

        width_x = x_max - x_min
        width_y = y_max - y_min
        center_x = x_min + width_x / tf.constant(2.0)
        center_y = y_min + width_y / tf.constant(2.0)

        geometry = tf.stack([center_x, center_y, width_x, width_y], axis=1)

        # Convert from pixel to normalized coordinates.
        image_width_height = image_shape[:2][::-1]
        image_width_height = tf.cast(image_width_height, tf.float32)
        image_width_height = tf.tile(image_width_height, (2,))
        return geometry / image_width_height

    # Handle the case where we have no detections.
    return tf.cond(
        tf.shape(bbox_coords)[0] > 0,
        _extract_features,
        lambda: tf.zeros((0, 4), dtype=tf.float32),
    )


def _extract_detection_images(
    *, bbox_coords: tf.Tensor, image: tf.Tensor, config: ModelConfig
) -> tf.Tensor:
    """
    Extracts detections from an image.

    Args:
        bbox_coords: The normalized bounding box coordinates. Should have
            the shape `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`.
        image: The input image to extract from. Should have shape
            `[height, width, num_channels]`.
        config: The model configuration to use.

    Returns:
        The extracted detections, resized to the specified detection size.
        Will have shape `[num_detections, d_height, d_width, num_channels]`.

    """
    bbox_coords = tf.ensure_shape(bbox_coords, (None, 4))

    # Convert from pixel to normalized coordinates.
    image_height_width = tf.shape(image)[:2]
    image_height_width = tf.cast(image_height_width, tf.float32)
    image_height_width = tf.tile(image_height_width, (2,))
    bbox_coords = bbox_coords / image_height_width

    # We only have one image...
    image = tf.expand_dims(image, axis=0)
    num_detections = tf.shape(bbox_coords)[0]
    box_indices = tf.zeros((num_detections,), dtype=tf.int32)

    detection_size = config.image_input_shape[:2]
    extracted = tf.image.crop_and_resize(
        image, bbox_coords, box_indices, crop_size=detection_size
    )
    # Make sure the result has the expected shape.
    extracted = tf.ensure_shape(extracted, (None,) + config.image_input_shape)

    # Convert back to uint8s.
    return tf.cast(extracted, tf.uint8)


def _extract_bbox_coords(feature_dict: Feature) -> tf.Tensor:
    """
    Extracts bounding box coordinates from a feature dictionary.

    Args:
        feature_dict: The raw (combined) feature dictionary for one image.

    Returns:
        The bounding boxes of the detections, in the form
        `[y_min, x_min, y_max, x_max]`.

    """
    x_min = feature_dict[Otf.OBJECT_BBOX_X_MIN.value]
    x_max = feature_dict[Otf.OBJECT_BBOX_X_MAX.value]
    y_min = feature_dict[Otf.OBJECT_BBOX_Y_MIN.value]
    y_max = feature_dict[Otf.OBJECT_BBOX_Y_MAX.value]
    return tf.stack([y_min, x_min, y_max, x_max], axis=1)


def _decode_image(feature_dict: Feature) -> tf.Tensor:
    """
    Decodes an image from a feature dictionary.

    Args:
        feature_dict: The raw (combined) feature dictionary for one image.

    Returns:
        The raw decoded image.

    """
    image_encoded = feature_dict[Otf.IMAGE_ENCODED.value]
    return tf.io.decode_jpeg(image_encoded[0])


def _make_heat_map_and_offsets(
    geometric_features: tf.Tensor,
    *,
    config: ModelConfig,
    image_shape: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates the detection heatmap and offsets tensor.

    Args:
        geometric_features: The geometric features for the image. Should have
            the form `[center_x, center_y, width, height]`.
        config: The model configuration.
        image_shape: The shape of the input image.

    Returns:
        The corresponding detection heatmap, and offset tensor. The latter will
        have the shape `[num_detections, width_offset, height_offset]`.

    """
    image_size = image_shape[:2][::-1]
    center_points = geometric_features[:, :2]
    down_sample_factor = tf.constant(config.detection_down_sample_factor)

    # Figure out the size of the heat maps.
    heat_map_size = image_size // down_sample_factor
    # Create the heat map.
    heat_map = make_heat_map(
        center_points,
        map_size=heat_map_size,
        sigma=config.detection_sigma,
    )

    # Compute the offset.
    center_points_px = center_points * tf.cast(image_size, tf.float32)
    offsets = (
        tf.cast(tf.round(center_points_px), tf.int32) % down_sample_factor
    )

    return heat_map, offsets


def _load_single_image_features(
    features: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_frame: bool = False,
    include_heat_map: bool = False,
    augmentation_config: DataAugmentationConfig = DataAugmentationConfig(),
) -> tf.data.Dataset:
    """
    Loads the features that can be extracted from a single image.

    Args:
        features: The dataset of feature dictionaries for each image.
        config: The model configuration to use.
        include_frame: If true, include the full frame image in the features.
        include_heat_map: Whether to include a detection heatmap.
        augmentation_config: Configuration for data augmentation.

    Returns:
        A dataset with elements that are dictionaries with the single-image
        features.

    """

    def _process_image(feature_dict: Feature) -> Feature:
        bbox_coords = _extract_bbox_coords(feature_dict)
        image = _decode_image(feature_dict)

        # Add random jitter.
        bbox_coords = _add_bbox_jitter(
            bbox_coords,
            image_shape=tf.shape(image),
            max_fractional_change=augmentation_config.max_bbox_jitter,
        )

        # Extract the detection crops.
        detections = _extract_detection_images(
            bbox_coords=bbox_coords, image=image, config=config
        )
        # Augment the crops.
        detections = _augment_images(detections, augmentation_config)

        # Compute the geometric features.
        geometric_features = _get_geometric_features(
            bbox_coords, image_shape=tf.shape(image)
        )

        object_ids = feature_dict[Otf.OBJECT_ID.value]
        frame_num = feature_dict[Otf.IMAGE_FRAME_NUM.value][0]
        sequence_id = feature_dict[Otf.IMAGE_SEQUENCE_ID.value][0]

        loaded_features = {
            FeatureName.DETECTIONS.value: detections,
            FeatureName.GEOMETRY.value: geometric_features,
            FeatureName.OBJECT_IDS.value: object_ids,
            FeatureName.FRAME_NUM.value: frame_num,
            FeatureName.SEQUENCE_ID.value: sequence_id,
        }
        if include_frame:
            loaded_features[FeatureName.FRAME_IMAGE.value] = image

        if include_heat_map:
            # Compute the detection heatmap and offsets.
            heatmap, offsets = _make_heat_map_and_offsets(
                geometric_features, config=config, image_shape=tf.shape(image)
            )
            loaded_features[FeatureName.HEAT_MAP.value] = heatmap
            loaded_features[FeatureName.DETECTIONS_OFFSETS.value] = offsets

        return loaded_features

    return features.map(_process_image, num_parallel_calls=_NUM_THREADS)


def _window_to_nested(windowed_features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Transforms a dataset that we have applied windowing to into one where
    each window is represented as a Tensor instead of a sub-dataset.

    Args:
        windowed_features: The windowed feature dataset.

    Returns:
        The same dataset, with windows represented as Tensors.

    """

    def _convert_element(features: MaybeRaggedFeature) -> MaybeRaggedFeature:
        # Windowing combines features into sub-datasets with two elements. To
        # access them, we will batch them into a single element and then
        # extract them.
        def _as_single_element(
            feature_key: str,
        ) -> Union[tf.Tensor, tf.RaggedTensor]:
            window_dataset = features[feature_key]
            window_dataset = window_dataset.apply(
                tf.data.experimental.dense_to_ragged_batch(2)
            )
            return tf.data.experimental.get_single_element(window_dataset)

        # Convert every feature.
        converted = {}
        for name in features:
            converted[name] = _as_single_element(name)

        return converted

    return windowed_features.map(_convert_element)


def _load_pair_features(features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Loads the features that need to be extracted from a consecutive image
    pair.

    Args:
        features: The single-image features for a batch of two consecutive
            frames.

    Returns:
        A dataset with elements that contain a dictionary of input features
        and a dictionary of target features for each frame pair.

    """

    def _process_pair(
        pair_features: Feature,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        object_ids = pair_features[FeatureName.OBJECT_IDS.value]
        detections = pair_features[FeatureName.DETECTIONS.value]
        geometry = pair_features[FeatureName.GEOMETRY.value]
        sequence_ids = pair_features[FeatureName.SEQUENCE_ID.value]
        frame_images = pair_features.get(FeatureName.FRAME_IMAGE.value, None)
        heat_maps = pair_features.get(FeatureName.HEAT_MAP.value, None)
        offsets = pair_features.get(FeatureName.DETECTIONS_OFFSETS.value, None)

        # Compute the ground-truth Sinkhorn matrix.
        tracklet_ids = object_ids[0]
        detection_ids = object_ids[1]
        sinkhorn = construct_gt_sinkhorn_matrix(
            detection_ids=detection_ids, tracklet_ids=tracklet_ids
        )
        # The sinkhorn matrix produced by the model is flattened.
        sinkhorn = tf.reshape(sinkhorn, (-1,))
        # Assignment target is the same as the sinkhorn matrix, just not a
        # float.
        assignment = tf.cast(sinkhorn, tf.bool)

        tracklets = detections[0]
        detections = detections[1]
        tracklet_geometry = geometry[0]
        detection_geometry = geometry[1]

        # Merge everything into input and target feature dictionaries.
        inputs = {
            ModelInputs.DETECTIONS.value: detections,
            ModelInputs.TRACKLETS.value: tracklets,
            ModelInputs.DETECTION_GEOMETRY.value: detection_geometry,
            ModelInputs.TRACKLET_GEOMETRY.value: tracklet_geometry,
            ModelInputs.SEQUENCE_ID.value: sequence_ids,
        }
        if frame_images is not None:
            # We only provide the current frame image.
            inputs[ModelInputs.DETECTIONS_FRAME.value] = frame_images[0]
        if heat_maps is not None:
            inputs[ModelInputs.DETECTIONS_HEATMAP.value] = heat_maps[0]
            inputs[ModelInputs.DETECTIONS_OFFSETS.value] = offsets[0]

        targets = {
            ModelTargets.SINKHORN.value: sinkhorn,
            ModelTargets.ASSIGNMENT.value: assignment,
        }

        return inputs, targets

    return features.map(_process_pair, num_parallel_calls=_NUM_THREADS)


def _filter_empty(features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Filters out examples that contain no detections at all.

    Args:
        features: The dataset containing input and target features.

    Returns:
        The same dataset, but with empty examples removed.

    """

    def _dense_shape(tensor: Union[tf.Tensor, tf.RaggedTensor]) -> tf.Tensor:
        if isinstance(tensor, tf.Tensor):
            return tf.shape(tensor)
        else:
            return tensor.bounding_shape()

    def _is_not_empty(inputs: MaybeRaggedFeature, _) -> tf.Tensor:
        # Check both detections and tracklets, and eliminate examples where
        # either is empty.
        detections = inputs[ModelInputs.DETECTIONS.value]
        tracklets = inputs[ModelInputs.TRACKLETS.value]

        # Get the shape.
        detections_shape = _dense_shape(detections)
        tracklets_shape = _dense_shape(tracklets)

        return tf.logical_and(detections_shape[0] > 0, tracklets_shape[0] > 0)

    return features.filter(_is_not_empty)


def _filter_out_of_order(pair_features: tf.data.Dataset) -> tf.data.Dataset:
    """
    A side-effect of repeating the input dataset multiple times is that
    the windowing will produce an invalid frame pair at the seams between
    the dataset. This transformation filters those out.

    Args:
        pair_features: The paired features.

    Returns:
        The filtered paired features.

    """

    def _is_ordered(_pair_features: Feature) -> bool:
        frame_nums = _pair_features[FeatureName.FRAME_NUM.value]
        # Compare frame numbers to ensure that these frames are ordered.
        first_frame_num = frame_nums[0]
        next_frame_num = frame_nums[1]

        return first_frame_num < next_frame_num

    return pair_features.filter(_is_ordered)


def _drop_mask(
    drop_probability: float = 0.5,
    repeat_probability: float = 0.9,
    width: int = 1,
) -> Iterable[np.ndarray]:
    """
    Generator that produces an infinite sequence of booleans indicating
    whether corresponding values should be dropped.

    Args:
        drop_probability: The probability of dropping the next item when we
            didn't drop the previous one.
        repeat_probability: The probability of dropping the next item when we
            did drop the previous one.
        width: The width of the mask to create.

    Yields:
        Boolean array of length _MASK_WIDTH where each element indicates
        whether a value should be kept.

    """
    generator = np.random.default_rng(_RANDOM_SEED)

    currently_dropping = np.zeros((width,), dtype=np.bool)
    while True:
        threshold = np.where(
            currently_dropping, repeat_probability, drop_probability
        )

        currently_dropping = generator.random(size=(width,)) < threshold
        yield np.logical_not(currently_dropping)


def _randomize_example_spacing(
    examples: tf.data.Dataset,
    *,
    drop_probability: float,
    repeats: int = 1,
) -> tf.data.Dataset:
    """
    Nominally, we create training examples from consecutive frames. However,
    better training results can sometimes be achieved by randomly skipping some
    frames. This transformation implements this using a geometric distribution
    to determine how many frames to skip.

    Args:
        examples: The dataset containing examples. This will not actually be
            interpreted, so it doesn't matter what it contains.
        drop_probability:
            The probability of dropping a particular frame. This is the p-value
            for the geometric distribution.
        repeats: Repeat the underlying dataset this many times. This allows us
            to make the length of the output similar to that of the input,
            despite dropping data.

    Returns:
        The transformed dataset, with some items dropped.

    """
    examples = examples.repeat(repeats)

    # Create the masks.
    drop_mask = partial(
        _drop_mask,
        drop_probability=drop_probability,
        repeat_probability=drop_probability,
    )
    drop_mask = tf.data.Dataset.from_generator(
        drop_mask, output_signature=tf.TensorSpec(shape=(1,), dtype=tf.bool)
    )
    # Combine the masks with the dataset.
    examples_with_masks = tf.data.Dataset.zip((examples, drop_mask))

    # Use the mask to perform filtering.
    filtered = examples_with_masks.filter(lambda _, m: tf.squeeze(m))

    # Remove the extraneous mask component.
    return filtered.map(lambda e, _: e, num_parallel_calls=_NUM_THREADS)


def _ensure_ragged(feature: MaybeRagged) -> tf.RaggedTensor:
    """
    Ensures that a tensor is ragged.

    Args:
        feature: The feature to check.

    Returns:
        The ragged feature.

    """
    if isinstance(feature, tf.RaggedTensor):
        # Already ragged.
        return feature
    # Otherwise, make it ragged.
    return tf.RaggedTensor.from_tensor(feature)


def _ensure_not_ragged(feature: MaybeRagged) -> tf.Tensor:
    """
    Ensures that a tensor is not ragged.

    Args:
        feature: The feature to check.

    Returns:
        The feature as a normal tensor, padded with zeros if necessary.

    """
    if isinstance(feature, tf.Tensor):
        # Already not ragged.
        return feature
    # Otherwise, make it ragged.
    return feature.to_tensor()


def _transform_features(
    features: tf.data.Dataset,
    *,
    transformer: Callable[[MaybeRagged], MaybeRagged],
    input_keys: Iterable[str] = (),
    target_keys: Iterable[str] = (),
) -> tf.data.Dataset:
    """
    Selectively transforms certain features in a dataset.

    Args:
        features: Dataset containing input and target feature dictionaries.
        transformer: Function that transforms a given tensor.
        input_keys: The keys in the input that we want to make ragged.
        target_keys: The keys in the targets that we want to make ragged.

    Returns:
        Dataset with the same features, but ensuring that the specified ones
        are ragged.

    """

    def _ensure_element_ragged(
        inputs: MaybeRaggedFeature, targets: MaybeRaggedFeature
    ) -> Tuple[MaybeRaggedFeature, MaybeRaggedFeature]:
        # Select only keys that exist in the data.
        existing_input_keys = frozenset(input_keys) & inputs.keys()
        existing_target_keys = frozenset(target_keys) & targets.keys()

        for key in existing_input_keys:
            inputs[key] = transformer(inputs[key])
        for key in existing_target_keys:
            targets[key] = transformer(targets[key])

        return inputs, targets

    return features.map(
        _ensure_element_ragged, num_parallel_calls=_NUM_THREADS
    )


def _inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_empty: bool = False,
    include_frame: bool = False,
    include_heat_map: bool = False,
    drop_probability: float = 0.0,
    repeats: int = 1,
    augmentation_config: DataAugmentationConfig = DataAugmentationConfig(),
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        config: Model configuration we are loading data for.
        include_empty: If true, will include examples with no detections
            or tracklets. Otherwise, it will filter them.
        include_frame: If true, will include the full frame image as well
            as the detection crops.
        include_heat_map: Whether to include a detection heatmap.
        drop_probability: Probability to drop a particular example.
        repeats: Number of times to repeat the dataset to make up for dropped
            examples.
        augmentation_config: Configuration to use for data augmentation.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    # Do the example filtering.
    filtered_raw = _randomize_example_spacing(
        raw_dataset, drop_probability=drop_probability, repeats=repeats
    )

    # Deserialize it.
    deserialized = filtered_raw.map(
        lambda s: tf.io.parse_single_example(s, _FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )

    # Extract the features.
    single_image_features = _load_single_image_features(
        deserialized,
        config=config,
        include_frame=include_frame,
        include_heat_map=include_heat_map,
        augmentation_config=augmentation_config,
    )
    # Break into pairs.
    image_pairs = _window_to_nested(
        single_image_features.window(2, shift=1, drop_remainder=True)
    )
    image_pairs = _filter_out_of_order(image_pairs)
    pair_features = _load_pair_features(image_pairs)

    # Remove empty examples.
    if not include_empty:
        pair_features = _filter_empty(pair_features)

    return pair_features


def _batch_and_prefetch(
    dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
) -> tf.data.Dataset:
    """
    Batches and prefetches data from a dataset.

    Args:
        dataset: The dataset to process.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        The batched dataset.

    """
    # Construct batches.
    batched = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size)
    )

    # `Dataset.map()` doesn't always correctly figure out which features
    # should be ragged, and which shouldn't be, so we ensure ourselves that
    # they are correct.
    ragged = _transform_features(
        batched,
        input_keys=_RAGGED_INPUTS,
        target_keys=_RAGGED_TARGETS,
        transformer=_ensure_ragged,
    )
    ragged = _transform_features(
        ragged,
        input_keys=_NON_RAGGED_INPUTS,
        target_keys=_NON_RAGGED_TARGETS,
        transformer=_ensure_not_ragged,
    )

    return ragged.prefetch(num_prefetch_batches)


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model, with batching and pre-fetching.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.
        kwargs: Will be forwarded to `_inputs_and_targets_from_dataset`.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    inputs_and_targets = _inputs_and_targets_from_dataset(
        raw_dataset, **kwargs
    )
    return _batch_and_prefetch(
        inputs_and_targets,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
    )


def inputs_and_targets_from_datasets(
    raw_datasets: Iterable[tf.data.Dataset],
    *,
    interleave: bool = True,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Deserializes and interleaves data from multiple datasets, and coerces it
    into the form used by the model.

    Args:
        raw_datasets: The raw datasets to draw from.
        interleave: Allows frames from multiple datasets to be interleaved
            with each-other if true. Set to false if you want to keep
            individual clips intact.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.
        **kwargs: Will be forwarded to `_inputs_and_targets_from_dataset`.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    # Parse all the data.
    parsed_datasets = []
    for raw_dataset in raw_datasets:
        parsed_datasets.append(
            _inputs_and_targets_from_dataset(raw_dataset, **kwargs)
        )

    if interleave:
        # Interleave the results.
        choices = tf.data.Dataset.range(len(parsed_datasets)).repeat()
        maybe_interleaved = tf.data.experimental.choose_from_datasets(
            parsed_datasets, choices
        )

    else:
        # Simply concatenate all of them together.
        maybe_interleaved = parsed_datasets[0]
        for dataset in parsed_datasets[1:]:
            maybe_interleaved = maybe_interleaved.concatenate(dataset)

    return _batch_and_prefetch(
        maybe_interleaved,
        batch_size=batch_size,
        num_prefetch_batches=num_prefetch_batches,
    )


def drop_detections(
    inputs_dataset: tf.data.Dataset,
    *,
    drop_probability: float,
    repeat_probability: float,
) -> tf.data.Dataset:
    """
    Modifies a dataset in order to drop random detections.

    Args:
        inputs_dataset: The dataset to modify, containing just the inputs. It
            is necessary that this dataset not be batched.
        drop_probability: The probability of dropping a detection given that
            it has not been dropped in the previous frame.
        repeat_probability: The probability of dropping a detection given
            that it has been dropped in the previous frame.

    Returns:
        The modified dataset.

    """
    # Width of the drop mask to generate. If there are more than this number
    # of detections, the mask will be repeated.
    _MASK_WIDTH = 8

    def _drop_mask() -> Iterable[np.ndarray]:
        """
        Generator that produces an infinite sequence of booleans indicating
        whether corresponding values should be dropped.

        Yields:
            Boolean array of length _MASK_WIDTH where each element indicates
            whether a value should be dropped.

        """
        currently_dropping = np.zeros((_MASK_WIDTH,), dtype=np.bool)
        while True:
            threshold = np.where(
                currently_dropping, repeat_probability, drop_probability
            )

            currently_dropping = np.random.rand(_MASK_WIDTH) < threshold
            yield currently_dropping

    # Combine the inputs with the drop mask.
    drop_mask = tf.data.Dataset.from_generator(
        _drop_mask, output_types=tf.bool
    )
    inputs_with_mask = tf.data.Dataset.zip((inputs_dataset, drop_mask))

    def _apply_mask(inputs: Feature, _drop_mask: tf.Tensor) -> Feature:
        detections = inputs[ModelInputs.DETECTIONS.value]
        geometry = inputs[ModelInputs.DETECTION_GEOMETRY.value]

        # Make sure the mask is the proper shape.
        num_detections = tf.shape(detections)[0]
        mask_multiples = num_detections // _MASK_WIDTH + 1
        mask_tiled = tf.tile(_drop_mask, tf.expand_dims(mask_multiples))
        mask_tiled = mask_tiled[:num_detections]

        # Flip the mask because it tells us what to drop, not keep.
        mask_flipped = tf.logical_not(mask_tiled)
        detections = tf.boolean_mask(detections, mask_flipped)
        geometry = tf.boolean_mask(geometry, mask_flipped)

        features = inputs
        features[ModelInputs.DETECTIONS.value] = detections
        features[ModelInputs.DETECTION_GEOMETRY.value] = geometry
        return features

    return inputs_with_mask.map(_apply_mask)
