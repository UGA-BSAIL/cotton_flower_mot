import enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, List

import numpy as np
import tensorflow as tf
from pydantic.dataclasses import dataclass

from .assignment import construct_gt_sinkhorn_matrix
from .color_utils import rgb_to_hcl
from .config import ModelConfig
from .heat_maps import make_object_heat_map
from .schemas import ColorizationTargets, ModelInputs, ModelTargets
from .schemas import ObjectTrackingFeatures as Otf
from .schemas import RotNetTargets
from .schemas import UnannotatedFeatures as Uf

_OTF_FEATURE_DESCRIPTION = {
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
    Otf.HEATMAP_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
}
"""
Descriptions of the features found in the dataset containing flower annotations.
"""

_UF_FEATURE_DESCRIPTION = {
    Uf.IMAGE_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Uf.IMAGE_FRAME_NUM.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Uf.IMAGE_SEQUENCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing unannotated images.
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
    ModelInputs.DETECTION_GEOMETRY.value,
    ModelInputs.TRACKLET_GEOMETRY.value,
}
"""
Input features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_INPUTS = {
    ModelInputs.SEQUENCE_ID.value,
    ModelInputs.DETECTIONS_FRAME.value,
    ModelInputs.TRACKLETS_FRAME.value,
}
"""
Input features that the model expects to be normal tensors.
"""
_RAGGED_TARGETS = {
    ModelTargets.GEOMETRY_DENSE_PRED.value,
    ModelTargets.GEOMETRY_SPARSE_PRED.value,
}
"""
Target features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_TARGETS = {
    ModelTargets.SINKHORN.value,
    ModelTargets.ASSIGNMENT.value,
    ModelTargets.HEATMAP.value,
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


@enum.unique
class HeatMapSource(enum.Enum):
    """
    Specifies how we want to produce heatmaps for CenterNet.
    """

    NONE = enum.auto()
    """
    Don't include heatmaps at all.
    """
    LOAD = enum.auto()
    """
    Load pre-generated heatmaps from the input dataset.
    """
    GENERATE = enum.auto()
    """
    Generate heatmaps on-the-fly.
    """


_NUM_THREADS = 4
"""
Number of threads to use for multi-threaded operations.
"""


@dataclass(frozen=True)
class DataAugmentationConfig:
    """
    Configuration to use for data augmentation.

    Attributes:
        max_brightness_delta: Maximum amount to adjust the brightness by.
        max_hue_delta: Maximum amount to adjust the hue by.

        min_contrast: Minimum contrast to use.
        max_contrast: Maximum contrast to use.

        min_saturation: Minimum saturation factor to use.
        max_saturation: Maximum saturation factor to use.

        max_bbox_jitter: Maximum amount of bounding box jitter to add, in
            fractions of a frame.
        false_positive_rate: The (simulated) false-positive rate to use for
            tracking.

        flip: Whether to allow horizontal and vertical flipping.
    """

    max_brightness_delta: float = 0.0
    max_hue_delta: float = 0.0

    min_contrast: Optional[float] = None
    max_contrast: Optional[float] = None

    min_saturation: Optional[float] = None
    max_saturation: Optional[float] = None

    max_bbox_jitter: float = 0.0
    false_positive_rate: float = 0.0

    flip: bool = False


def _flip_geometry(
    geometry: tf.Tensor, *, left_right: tf.Tensor, up_down: tf.Tensor
) -> tf.Tensor:
    """
    Flips the bounding box geometry about the central axes of an image.

    Args:
        geometry: The bounding box geometry, in the form
            `[center_x, center_y, ...]`. No attributes beyond the center
            point will be changed.
        left_right: 0D bool tensor, whether to flip horizontally.
        up_down: 0D bool tensor, whether to flip vertically.

    Returns:
        The flipped geometry.

    """
    left_right = tf.ensure_shape(left_right, ())
    up_down = tf.ensure_shape(up_down, ())

    center_points = geometry[:, :2]
    other_attributes = geometry[:, 2:]

    def flip_up_down() -> tf.Tensor:
        center_distance = center_points - tf.constant([0.0, 0.5])
        flipped_distance = center_distance * tf.constant([1.0, -1.0])
        return flipped_distance + tf.constant([0.0, 0.5])

    def flip_left_right() -> tf.Tensor:
        center_distance = center_points - tf.constant([0.5, 0.0])
        flipped_distance = center_distance * tf.constant([-1.0, 1.0])
        return flipped_distance + tf.constant([0.5, 0.0])

    # Geometry should be normalized, so we can just flip about 0.5.
    center_points = tf.cond(up_down, flip_up_down, lambda: center_points)
    center_points = tf.cond(left_right, flip_left_right, lambda: center_points)
    return tf.concat((center_points, other_attributes), axis=1)


def _random_flip(
    *,
    images: Iterable[tf.Tensor],
    heatmaps: Optional[Iterable[tf.Tensor]],
    geometry: Iterable[tf.Tensor],
) -> Tuple[List[tf.Tensor], Optional[List[tf.Tensor]], List[tf.Tensor],]:
    """
    Randomly flips input images vertically and horizontally,
    also transforming the corresponding geometry. It will apply the same
    transformation to pairs of images, in order to avoid disturbing the
    geometric relationships that the tracker relies on.

    Args:
        images: The 3D images to possibly flip.
        heatmaps: The corresponding heatmaps, or None if there are no heatmaps.
        geometry: Bounding box geometry for the images, of the form
            `[center_x, center_y, width, height, offset_x, offset_y]`.

    Returns:
        The same images, heatmaps, and geometry, possibly flipped.

    """

    def _do_flip(_image: tf.Tensor) -> tf.Tensor:
        # Flip the image and heatmap.
        _image = tf.cond(
            should_flip_lr,
            lambda: tf.image.flip_left_right(_image),
            lambda: _image,
        )
        _image = tf.cond(
            should_flip_ud,
            lambda: tf.image.flip_up_down(_image),
            lambda: _image,
        )

        return _image

    # Determine if we should do the flipping.
    should_flip = tf.random.uniform((2,), maxval=2, dtype=tf.int32)
    should_flip = tf.cast(should_flip, tf.bool)
    should_flip_lr = should_flip[0]
    should_flip_ud = should_flip[1]

    images = [_do_flip(i) for i in images]
    if heatmaps is not None:
        heatmaps = [_do_flip(h) for h in heatmaps]

    # Flip the bounding boxes.
    _flip_geo_ = partial(
        _flip_geometry, left_right=should_flip_lr, up_down=should_flip_ud
    )
    geometry = [_flip_geo_(g) for g in geometry]

    return images, heatmaps, geometry


def _add_bbox_jitter(
    bbox_coords: tf.Tensor,
    max_fractional_change: float = 0.05,
) -> tf.Tensor:
    """
    Adds uniform random jitter to bounding box coordinates.
    Args:
        bbox_coords: The bounding box coordinates. Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`, in pixels.
        max_fractional_change: The maximum absolute amount that a coordinate
            can change, as a fraction of the original coordinate value.
    Returns:
        The bounding box coordinates with jitter.
    """
    # The maximum change will be the same for the center x, center y,
    # width and height dimensions.
    max_fractional_change = tf.constant([max_fractional_change] * 4)

    # Generate random jitters.
    jitter_magnitude = tf.random.uniform(
        tf.shape(bbox_coords),
        minval=-max_fractional_change,
        maxval=max_fractional_change,
    )

    # Apply the jitters.
    return bbox_coords + jitter_magnitude


def _add_random_bboxes(
    bbox_coords: tf.Tensor,
    *,
    rate: float,
) -> tf.Tensor:
    """
    Adds random bounding boxes to a list of bounding boxes.

    Args:
        bbox_coords: The bounding box coordinates. Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`, in pixels.
        rate: The false-positive rate we would like to simulate.

    Returns:
        The same bounding boxes, with any false positives appended, and the
        modified Sinkhorn matrix.

    """
    # Determine number of false positives.
    num_false_positives = tf.random.poisson((1,), rate, dtype=tf.int32)

    # Generate random bounding boxes.
    fp_shape = tf.concat((num_false_positives, [4]), axis=0)
    fp_boxes = tf.random.uniform(
        fp_shape, minval=0.0, maxval=1.0, dtype=bbox_coords.dtype
    )

    return tf.concat((bbox_coords, fp_boxes), axis=0)


def _augment_with_false_positives(
    *,
    tracklets_bbox_coords: tf.Tensor,
    detections_bbox_coords: tf.Tensor,
    sinkhorn: tf.Tensor,
    rate: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Adds (simulated) false positives to the input data for tracking. This can
    help to train a more robust tracker.

    Args:
        tracklets_bbox_coords: The bounding box coordinates for the tracklets.
            Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`, in pixels.
        detections_bbox_coords: The bounding box coordinates for the detections.
        sinkhorn: The ground-truth sinkhorn matrix. Will be modified with any
            false positives. Should have a shape of
            `[num_tracklets, num_detections]`.
        rate: The false-positive rate we would like to simulate.

    Returns:
        The tracklet bounding boxes, the detection bounding boxes, and the
        modified Sinkhorn matrix.

    """
    # Add random bounding boxes.
    tracklets_bbox_coords = _add_random_bboxes(
        tracklets_bbox_coords, rate=rate
    )
    detections_bbox_coords = _add_random_bboxes(
        detections_bbox_coords, rate=rate
    )

    # Pad the Sinkhorn matrix to reflect the false positives.
    padded_size = tf.stack(
        (
            tf.shape(tracklets_bbox_coords)[0],
            tf.shape(detections_bbox_coords)[0],
        )
    )
    needed_padding = tf.shape(sinkhorn) - padded_size
    sinkhorn = tf.pad(
        sinkhorn, ((0, needed_padding[0]), (0, needed_padding[1]))
    )

    return tracklets_bbox_coords, detections_bbox_coords, sinkhorn


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


def _augment_inputs(
    *,
    images: Iterable[tf.Tensor],
    heatmaps: Optional[Iterable[tf.Tensor]],
    input_geometry: Iterable[tf.Tensor],
    target_geometry: Iterable[tf.Tensor],
    config: DataAugmentationConfig,
) -> Tuple[
    List[tf.Tensor],
    Optional[List[tf.Tensor]],
    List[tf.Tensor],
    List[tf.Tensor],
]:
    """
    Applies data augmentation to images.

    Args:
        images: The images to augment.
        heatmaps: The corresponding heatmaps, or None if there are no heatmaps.
        input_geometry: Bounding box geometry, of the form
            `[center_x, center_y, width, height]`, used as input for the
            tracker.
        target_geometry: Bounding box geometry, of the form
            `[center_x, center_y, width, height, offset_x, offset_y]`,
            used as targets for the detector.
        config: Configuration for data augmentation.

    Returns:
        The augmented images, heatmaps, input geometry, and target geometry.

    """
    images = [_augment_images(i, config) for i in images]

    # Perform flipping.
    if config.flip:
        input_geometry = list(input_geometry)
        target_geometry = list(target_geometry)
        all_geometry = input_geometry + target_geometry

        images, heatmaps, all_geometry = _random_flip(
            images=images, heatmaps=heatmaps, geometry=all_geometry
        )

        # Separate the geometry again.
        num_inputs = len(input_geometry)
        input_geometry = all_geometry[:num_inputs]
        target_geometry = all_geometry[num_inputs:]

    # Apply random jitter and false-positives, but only to the tracker inputs.
    input_geometry = [
        _add_bbox_jitter(
            g,
            max_fractional_change=config.max_bbox_jitter,
        )
        for g in input_geometry
    ]

    return images, heatmaps, input_geometry, target_geometry


def _extract_rotations(image: tf.Tensor) -> tf.Tensor:
    """
    Extracts all four rotations from an image.

    Args:
        image: The image to extract rotations from.

    Returns:
        The four rotations, in a single tensor, with a new dimension
        prepended. The order is always 0, 90, 180, 270.

    """
    # 90-degree rotation.
    deg_90 = tf.image.rot90(image)
    # 180-degree rotation, which is just flipping.
    deg_180 = tf.image.flip_up_down(image)
    # 270-degree rotation.
    deg_270 = tf.image.flip_left_right(deg_90)

    return tf.stack([image, deg_90, deg_180, deg_270])


def _load_rot_net(
    unannotated_features: Feature, *, config: ModelConfig
) -> Tuple[Feature, Feature]:
    """
    Extracts RotNet features from the unannotated TFRecords.

    Args:
        unannotated_features: The raw unannotated features.
        config: The model configuration to use.

    Returns:
        Equivalent RotNet features. Will return input and target features.

    """
    # Decode the image.
    image_encoded = unannotated_features[Uf.IMAGE_ENCODED.value]
    image = _decode_image(image_encoded, ratio=2)

    # Crop to the RotNet input shape.
    image = tf.image.random_crop(image, size=config.rot_net_input_shape)

    # Labels are just integers.
    labels = tf.range(4)
    rotations = _extract_rotations(image)
    return {ModelInputs.DETECTIONS_FRAME.value: rotations}, {
        RotNetTargets.ROTATION_CLASS.value: labels,
    }


def _compute_histograms(
    image_channel: tf.Tensor,
    window_size: int = 7,
    num_bins: int = 32,
) -> tf.Tensor:
    """
    Computes the histogram of an image, using color information
    from a window around each pixel.

    Args:
        image_channel: The single-channel image to compute histograms for.
            Should be rank 2.
        window_size: The size of the window to use, in pixels.
        num_bins: The number of bins to use for the histogram.

    Returns:
        The histograms it computed. Will be a 3D matrix of shape
        [num_rows, num_columns, num_bins], containing the histograms for each
        input pixel.

    """
    input_shape = tf.shape(image_channel)
    image_channel = tf.expand_dims(tf.expand_dims(image_channel, 0), -1)

    # First, pre-bin all the values in the image.
    binned = tf.histogram_fixed_width_bins(
        image_channel, value_range=[0.0, 1.0], nbins=num_bins, dtype=tf.uint8
    )

    # Extract patches.
    patches = tf.image.extract_patches(
        binned,
        sizes=[1, window_size, window_size, 1],
        strides=[1] * 4,
        rates=[1] * 4,
        padding="SAME",
    )
    # There will be only one batch dimension, so get rid of it. We also
    # flatten everything into a single matrix where each row is a patch.
    num_patches = tf.reduce_prod(patches.shape[:3])
    patches = tf.reshape(patches, tf.stack((num_patches, -1)))

    # Compute histograms of each patch.
    patches_sorted = tf.sort(patches, axis=1)
    histograms = tf.math.bincount(patches_sorted, axis=-1, minlength=num_bins)

    # Reshape to correspond with the image.
    return tf.reshape(histograms, tf.concat((input_shape, [-1]), axis=0))


def _load_colorization(
    unannotated_features: Feature, *, config: ModelConfig
) -> Tuple[Feature, Feature]:
    """
    Extracts colorization features from the unannotated TFRecords.

    Args:
        unannotated_features: The raw unannotated features.
        config: The model configuration to use.

    Returns:
        Equivalent Colorization features. Will return input and target features.

    """
    # Decode the image.
    image_encoded = unannotated_features[Uf.IMAGE_ENCODED.value]
    image = _decode_image(image_encoded, ratio=2)

    # Crop to the colorization input shape.
    input_shape_color = config.colorization_input_shape[:2] + (3,)
    image = tf.image.random_crop(image, size=input_shape_color)
    image_hcl = rgb_to_hcl(image)

    # Make sure the output has the correct shape.
    height, width, num_bins = config.colorization_output_shape
    image_hcl_small = tf.image.resize(image_hcl, (height, width))
    # Generate histograms.
    hue_histograms = _compute_histograms(
        image_hcl_small[:, :, 0], num_bins=num_bins
    )
    chroma_histograms = _compute_histograms(
        image_hcl_small[:, :, 1], num_bins=num_bins
    )

    lightness = image_hcl[:, :, 2]
    lightness = tf.cast(lightness * 255, tf.uint8)
    lightness = tf.expand_dims(lightness, axis=-1)
    return {ModelInputs.DETECTIONS_FRAME.value: lightness}, {
        ColorizationTargets.HUE_HIST.value: hue_histograms,
        ColorizationTargets.CHROMA_HIST.value: chroma_histograms,
    }


def _get_geometric_features(
    bbox_coords: tf.Tensor,
    *,
    image_shape: tf.Tensor,
    config: ModelConfig,
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
        config: The model configuration.

    Returns:
        A tensor of shape [N, 4], where N is the total number of bounding
        boxes in the input. The ith row specifies the normalized coordinates of
        the ith bounding box in the form
        `[center_x, center_y, width, height, offset_x, offset_y]`. Offsets are
        the calculated offsets for detection.

    """
    bbox_coords = tf.ensure_shape(bbox_coords, (None, 4))

    def _extract_features() -> tf.Tensor:
        x_min = bbox_coords[:, 1]
        x_max = bbox_coords[:, 3]
        y_min = bbox_coords[:, 0]
        y_max = bbox_coords[:, 2]

        size_x = x_max - x_min
        size_y = y_max - y_min
        center_x = x_min + size_x / tf.constant(2.0)
        center_y = y_min + size_y / tf.constant(2.0)

        image_width_height = image_shape[:2][::-1]
        image_width_height = tf.cast(image_width_height, tf.float32)

        # Compute the offset.
        center_points_px = tf.stack([center_x, center_y], axis=1)
        down_sample_factor = tf.constant(2**config.num_reduction_stages)
        offsets = (
            tf.cast(tf.round(center_points_px), tf.int32) % down_sample_factor
        )
        offsets = tf.cast(offsets, tf.float32)

        size_px = tf.stack([size_x, size_y], axis=1)
        geometry = tf.concat([center_points_px, size_px, offsets], axis=1)
        image_width_height_tiled = tf.tile(image_width_height, (3,))
        return geometry / image_width_height_tiled

    # Handle the case where we have no detections.
    return tf.cond(
        tf.shape(bbox_coords)[0] > 0,
        _extract_features,
        lambda: tf.zeros((0, 6), dtype=tf.float32),
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


def _decode_image(encoded: tf.Tensor, ratio: int = 1) -> tf.Tensor:
    """
    Decodes an image from a feature dictionary.

    Args:
        encoded: The encoded image.
        ratio: Downsample ratio to use when decoding.

    Returns:
        The raw decoded image.

    """
    return tf.io.decode_jpeg(encoded[0], ratio=ratio)


def _decode_heat_map(encoded: tf.Tensor) -> tf.Tensor:
    """
    Decodes the heat map from a feature dictionary.

    Args:
        encoded: The encoded heatmap.

    Returns:
        The raw decoded heatmap.

    """
    heat_map = tf.io.decode_png(encoded[0], dtype=tf.uint16)
    # Heat map is stored as ints, but we need it to be normalized floats.
    return tf.cast(heat_map, tf.float32) / np.iinfo(np.uint16).max


def _load_single_image_features(
    features: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_frame: bool = False,
    heat_map_source: HeatMapSource = HeatMapSource.LOAD,
) -> tf.data.Dataset:
    """
    Loads the features that can be extracted from a single image.

    Args:
        features: The dataset of feature dictionaries for each image.
        config: The model configuration to use.
        include_frame: If true, include the full frame image in the features.
        heat_map_source: Where to get the detection heatmap from.

    Returns:
        A dataset with elements that are dictionaries with the single-image
        features.

    """

    def _process_image(feature_dict: Feature) -> Feature:
        bbox_coords = _extract_bbox_coords(feature_dict)

        # Compute the geometric features.
        image = feature_dict[Otf.IMAGE_ENCODED.value]
        image_shape = tf.io.extract_jpeg_shape(image[0])
        geometric_features = _get_geometric_features(
            bbox_coords, image_shape=image_shape, config=config
        )

        object_ids = feature_dict[Otf.OBJECT_ID.value]
        frame_num = feature_dict[Otf.IMAGE_FRAME_NUM.value][0]
        sequence_id = feature_dict[Otf.IMAGE_SEQUENCE_ID.value][0]

        loaded_features = {
            FeatureName.GEOMETRY.value: geometric_features,
            FeatureName.OBJECT_IDS.value: object_ids,
            FeatureName.FRAME_NUM.value: frame_num,
            FeatureName.SEQUENCE_ID.value: sequence_id,
        }
        if include_frame:
            loaded_features[FeatureName.FRAME_IMAGE.value] = image
        if heat_map_source == HeatMapSource.LOAD:
            loaded_features[FeatureName.HEAT_MAP.value] = feature_dict[
                Otf.HEATMAP_ENCODED.value
            ]

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


def _load_pair_features(
    features: tf.data.Dataset,
    augmentation_config: DataAugmentationConfig = DataAugmentationConfig(),
) -> tf.data.Dataset:
    """
    Loads the features that need to be extracted from a consecutive image
    pair.

    Args:
        features: The single-image features for a batch of two consecutive
            frames.
        augmentation_config: The configuration to use for data augmentation.

    Returns:
        A dataset with elements that contain a dictionary of input features
        and a dictionary of target features for each frame pair.

    """

    def _process_pair(
        pair_features: Feature,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        object_ids = pair_features[FeatureName.OBJECT_IDS.value]
        geometry = pair_features[FeatureName.GEOMETRY.value]
        sequence_ids = pair_features[FeatureName.SEQUENCE_ID.value]
        frame_images = pair_features.get(FeatureName.FRAME_IMAGE.value, None)
        heat_maps = pair_features.get(FeatureName.HEAT_MAP.value, None)

        # Compute the ground-truth Sinkhorn matrix.
        tracklet_ids = object_ids[0]
        detection_ids = object_ids[1]
        sinkhorn = construct_gt_sinkhorn_matrix(
            detection_ids=detection_ids, tracklet_ids=tracklet_ids
        )

        tracklet_geometry = geometry[0]
        detection_geometry = geometry[1]

        # Compute augmented geometry for tracking.
        (
            augmented_tracklets,
            augmented_detections,
            sinkhorn,
        ) = _augment_with_false_positives(
            # For tracking, we don't need the offsets in the geometric features.
            tracklets_bbox_coords=tracklet_geometry[:, :4],
            detections_bbox_coords=detection_geometry[:, :4],
            sinkhorn=sinkhorn,
            rate=augmentation_config.false_positive_rate,
        )

        # The sinkhorn matrix produced by the model is flattened.
        sinkhorn = tf.reshape(sinkhorn, (-1,))
        # Assignment target is the same as the sinkhorn matrix, just not a
        # float.
        assignment = tf.cast(sinkhorn, tf.bool)

        # Merge everything into input and target feature dictionaries.
        inputs = {
            ModelInputs.DETECTION_GEOMETRY.value: augmented_detections,
            ModelInputs.TRACKLET_GEOMETRY.value: augmented_tracklets,
            ModelInputs.SEQUENCE_ID.value: sequence_ids,
        }
        if frame_images is not None:
            inputs[ModelInputs.TRACKLETS_FRAME.value] = frame_images[0]
            inputs[ModelInputs.DETECTIONS_FRAME.value] = frame_images[1]

        targets = {
            ModelTargets.SINKHORN.value: sinkhorn,
            ModelTargets.ASSIGNMENT.value: assignment,
            ModelTargets.GEOMETRY_DENSE_PRED.value: detection_geometry,
            ModelTargets.GEOMETRY_SPARSE_PRED.value: detection_geometry,
        }
        if heat_maps is not None:
            # We only provide the current frame heatmap.
            targets[ModelTargets.HEATMAP.value] = heat_maps[1]

        return inputs, targets

    return features.map(_process_pair, num_parallel_calls=_NUM_THREADS)


def _decode_images(
    features: tf.data.Dataset,
    *,
    model_config: ModelConfig,
    augmentation_config: DataAugmentationConfig = DataAugmentationConfig(),
    heat_map_source: HeatMapSource = HeatMapSource.LOAD,
) -> tf.data.Dataset:
    """
    Decodes images from pair features and does whatever processing has
    been deferred while the images were encoded. Since this is the slowest
    and most memory-intensive step of the entire pipeline, it makes sense to
    do this as late as possible after we've had a chance to filter out
    unnecessary data.

    Args:
        features: The pair features, with images still encoded.
        model_config: The model configuration to use.
        augmentation_config: Configuration for data augmentation.
        heat_map_source: Where we should get the heatmaps from.

    Returns:
        The same pair features, with the images decoded and processed.

    """

    def _process_example(
        inputs: Feature, targets: Feature
    ) -> Tuple[Feature, Feature]:
        # Get the image features.
        detections_frame = inputs.get(ModelInputs.DETECTIONS_FRAME.value, None)
        tracklets_frame = inputs.get(ModelInputs.TRACKLETS_FRAME.value, None)
        heatmap = targets.get(ModelTargets.HEATMAP.value, None)
        detections_geometry = inputs[ModelInputs.DETECTION_GEOMETRY.value]
        tracklets_geometry = inputs[ModelInputs.TRACKLET_GEOMETRY.value]
        target_geometry = targets[ModelTargets.GEOMETRY_DENSE_PRED.value]

        # Decode the image features.
        if heat_map_source == HeatMapSource.LOAD:
            # Heatmap should have been provided in this case.
            assert heatmap is not None
            heatmap = _decode_heat_map(heatmap)
        elif heat_map_source == HeatMapSource.GENERATE:
            # Generate a new heatmap.
            heatmap = make_object_heat_map(
                target_geometry[:, :4],
                map_size=tf.constant(model_config.heatmap_size),
                normalized=False,
            )

        if detections_frame is not None:
            detections_frame = _decode_image(detections_frame, ratio=2)
            tracklets_frame = _decode_image(tracklets_frame, ratio=2)

            # Perform data augmentation.
            (
                (detections_frame, tracklets_frame),
                (heatmap,),
                (detections_geometry, tracklets_geometry),
                (target_geometry,),
            ) = _augment_inputs(
                images=(detections_frame, tracklets_frame),
                heatmaps=(heatmap,),
                input_geometry=(
                    detections_geometry,
                    tracklets_geometry,
                ),
                target_geometry=(target_geometry,),
                config=augmentation_config,
            )

            inputs[ModelInputs.DETECTIONS_FRAME.value] = detections_frame
            inputs[ModelInputs.TRACKLETS_FRAME.value] = tracklets_frame
            # Update all the geometry.
            inputs[ModelInputs.DETECTION_GEOMETRY.value] = detections_geometry
            inputs[ModelInputs.TRACKLET_GEOMETRY.value] = tracklets_geometry
            targets[ModelTargets.GEOMETRY_DENSE_PRED.value] = target_geometry
            targets[ModelTargets.GEOMETRY_SPARSE_PRED.value] = target_geometry

        if heatmap is not None:
            targets[ModelTargets.HEATMAP.value] = heatmap

        return inputs, targets

    return features.map(_process_example, num_parallel_calls=_NUM_THREADS)


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
        detections = inputs[ModelInputs.DETECTION_GEOMETRY.value]
        tracklets = inputs[ModelInputs.TRACKLET_GEOMETRY.value]

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

    currently_dropping = np.zeros((width,), dtype=bool)
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

    def _apply_transform(
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

    return features.map(_apply_transform, num_parallel_calls=_NUM_THREADS)


def _inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_empty: bool = False,
    include_frame: bool = False,
    heat_map_source: HeatMapSource = HeatMapSource.LOAD,
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
        heat_map_source: Source to use for detection heatmap.
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
        lambda s: tf.io.parse_single_example(s, _OTF_FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )

    # Extract the features.
    single_image_features = _load_single_image_features(
        deserialized,
        config=config,
        include_frame=include_frame,
        heat_map_source=heat_map_source,
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

    # Decode the images.
    pair_features = _decode_images(
        pair_features,
        augmentation_config=augmentation_config,
        model_config=config,
        heat_map_source=heat_map_source,
    )

    return pair_features


def _batch_and_prefetch(
    dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 1,
    shuffle_buffer_size: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Batches and prefetches data from a dataset.

    Args:
        dataset: The dataset to process.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.
        shuffle_buffer_size: The buffer size to use for shuffling. If set to
            None, it will not shuffle at all.

    Returns:
        The batched dataset.

    """
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(batch_size * 20)

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

    prefetched = ragged.prefetch(num_prefetch_batches)

    options = tf.data.Options()
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    return prefetched.with_options(options)


def _shuffle_and_deserialize(
    dataset: tf.data.Dataset, num_shards: int = 100
) -> tf.data.Dataset:
    """
    Shuffles a dataset, and deserializes TFRecords.

    Args:
        dataset: The dataset to process.
        num_shards: The number of shards to use when randomizing. A larger
            number improves randomness, but may risk exhausting the maximum
            number of open files.

    Returns:
        The processed dataset.

    """
    # Shuffle all the clips together through sharding.
    sharded_datasets = []
    for i in range(0, num_shards):
        sharded_datasets.append(dataset.shard(num_shards, i))
    shuffled_dataset = tf.data.Dataset.sample_from_datasets(sharded_datasets)

    # Deserialize it.
    return shuffled_dataset.map(
        lambda s: tf.io.parse_single_example(s, _UF_FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )


def rot_net_inputs_and_targets_from_dataset(
    unannotated_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    batch_size: int = 8,
    num_prefetch_batches: int = 1,
) -> tf.data.Dataset:
    """
    Loads RotNet input features from a dataset of unannotated images.

    Args:
        unannotated_dataset: The dataset of unannotated images.
        config: Model configuration we are loading data for.
        batch_size: The batch size to use. The actual batch size will be
            multiplied by four. because it will include all four rotations.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        A dataset that produces input images and target classes.

    """
    deserialized = _shuffle_and_deserialize(unannotated_dataset)

    # Extract the rotations.
    rot_net_dataset = deserialized.map(
        partial(_load_rot_net, config=config), num_parallel_calls=_NUM_THREADS
    )
    # Unbatch to separate all the rotations.
    rot_net_dataset = rot_net_dataset.unbatch()

    # Batch and prefetch.
    batched = rot_net_dataset.batch(batch_size * 4)
    return batched.prefetch(num_prefetch_batches)


def colorization_inputs_and_targets_from_dataset(
    unannotated_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    batch_size: int = 8,
    num_prefetch_batches: int = 1,
) -> tf.data.Dataset:
    """
    Loads colorization input features from a dataset of unannotated images.

    Args:
        unannotated_dataset: The dataset of unannotated images.
        config: Model configuration we are loading data for.
        batch_size: The batch size to use. The actual batch size will be
            multiplied by four. because it will include all four rotations.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        A dataset that produces input images and target classes.

    """
    deserialized = _shuffle_and_deserialize(unannotated_dataset)

    # Extract the colorization features.
    rot_net_dataset = deserialized.map(
        partial(_load_colorization, config=config),
        num_parallel_calls=_NUM_THREADS,
    )

    # Batch and prefetch.
    batched = rot_net_dataset.batch(batch_size)
    return batched.prefetch(num_prefetch_batches)


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 1,
    shuffle_buffer_size: Optional[int] = None,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model, with batching and pre-fetching.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.
        shuffle_buffer_size: The buffer size to use for shuffling. If set to
            None, it will not shuffle at all.
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
        shuffle_buffer_size=shuffle_buffer_size,
    )


def inputs_and_targets_from_datasets(
    raw_datasets: Iterable[tf.data.Dataset],
    *,
    interleave: bool = True,
    batch_size: int = 32,
    num_prefetch_batches: int = 1,
    shuffle_buffer_size: Optional[int] = None,
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
        shuffle_buffer_size: The buffer size to use for shuffling. If set to
            None, it will not shuffle at all.
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
        maybe_interleaved = tf.data.Dataset.sample_from_datasets(
            parsed_datasets
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
        shuffle_buffer_size=shuffle_buffer_size,
    )
