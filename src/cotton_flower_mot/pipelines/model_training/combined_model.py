"""
Implements a combined detection + tracking model.
"""


from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .centernet_model import build_detection_model
from .gcnn_model import compute_association
from .layers.pooling import RoiPooling
from .layers.utility import BnActConv


def _extract_appearance_features(
    *,
    bbox_geometry: tf.RaggedTensor,
    image_features: tf.Tensor,
    config: ModelConfig
) -> tf.RaggedTensor:
    """
    Extracts the appearance features for the detections or tracklets.

    Args:
        bbox_geometry: The bounding box information. Should have
            shape `[batch_size, num_boxes, 4]`, where the second dimension is
            ragged, and the third is ordered `[x, y, width, height]`.
            tracklets.
        image_features: The raw image features from the detector.
        config: The model configuration to use.

    Returns:
        The extracted appearance features. They are a `RaggedTensor`
        with the shape `[batch_size, n_nodes, n_features]`, where the second
        dimension is ragged.

    """
    image_features = BnActConv(8, 1, activation="relu")(image_features)
    feature_crops = RoiPooling(config.roi_pooling_size)(
        (image_features, bbox_geometry)
    )

    # Coerce the features to the correct shape.
    def _flatten_features(_features: tf.RaggedTensor) -> tf.RaggedTensor:
        # We have to go through this annoying process to ensure that the
        # static shape remains correct.
        inner_shape = _features.shape[-3:]
        num_flat_features = np.prod(inner_shape)
        flat_features = tf.reshape(_features.values, (-1, num_flat_features))
        return _features.with_values(flat_features)

    features = layers.Lambda(_flatten_features)(feature_crops)
    return features


def _apply_detector(
    detector: tf.keras.Model,
    *,
    frames: tf.Tensor,
    confidence_threshold: float = 0.0
) -> Tuple[tf.Tensor, tf.Tensor, tf.RaggedTensor]:
    """
    Applies the detector model to an input.

    Args:
        detector: The detector model.
        frames: The input frames.
        confidence_threshold: The minimum confidence of detections. Any
            detections with lower confidence will be removed.

    Returns:
        The heatmaps, dense geometry predictions, and bounding boxes.

    """
    heatmap, dense_geometry, bboxes = detector(frames)

    # Remove low-confidence detections.
    confidence = bboxes[:, :, 4]
    bboxes = tf.ragged.boolean_mask(bboxes, confidence >= confidence_threshold)

    # Ensure that the resulting layers have the correct names when we set
    # them as outputs.
    heatmap = layers.Activation(
        "linear", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(heatmap)
    dense_geometry = layers.Activation(
        "linear", name=ModelTargets.GEOMETRY_DENSE_PRED.value, dtype=tf.float32
    )(dense_geometry)
    bboxes = layers.Activation(
        "linear",
        name=ModelTargets.GEOMETRY_SPARSE_PRED.value,
        dtype=tf.float32,
    )(bboxes)

    return heatmap, dense_geometry, bboxes


def _choose_rois_for_tracker(
    *,
    detection_bboxes: tf.RaggedTensor,
    tracker_roi_input: tf.RaggedTensor,
    use_gt_detections: tf.Tensor
) -> tf.RaggedTensor:
    """
    The tracker can operate in two modes, which are dynamically selectable at
    runtime. In training mode, ground-truth bounding boxes for objects to
    track are fed into the model, and these are used to train the tracker. In
    inference mode, however, we don't have any ground-truth, so we rely
    directly on the output of the detector.

    This function implements the logic necessary to select the proper source
    for these tracker ROIs. This is much more complicated than you would
    expect, mainly due to TensorFlow's obstreperousness about normal tensors vs.
    Keras tensors, and some (probable) bugs in the implementation of the
    Keras `Lambda` layer related to the handling of `RaggedTensor`s.

    Args:
        detection_bboxes: The bounding boxes from the detector. Should have
            the shape `[batch_size, (num_detections), 4]`.
        tracker_roi_input: The ground-truth bounding boxes input by the user.
            Should have the shape `[batch_size, (num_detections), 4]`.
        use_gt_detections: A special input that selects whether we use the
            detected or ground-truth bounding boxes. Should be a vector tensor
            with exactly one boolean element.

    Returns:
        The ROIs to use for the tracker, with the shape
        `[batch_size, (num_detections), 4]`.

    """
    # Keras Lambda layers seem to have some bugs when handling RaggedTensors,
    # so we convert them to normal tensors for this operation, and then back.
    roi_input_non_ragged = (
        tracker_roi_input.flat_values,
        tracker_roi_input.nested_row_splits,
    )
    detection_bboxes_non_ragged = (
        detection_bboxes.flat_values,
        detection_bboxes.nested_row_splits,
    )
    # If we are in training mode, use the input ROIs. Otherwise, just grab
    # the bounding boxes from the detector.
    detection_geometry_values, detection_geometry_splits = layers.Lambda(
        lambda params: tf.cond(
            params[0][0],
            # Use the ground-truth ROIs supplied by the user.
            lambda: params[1],
            # Use the detections.
            lambda: params[2],
        ),
        name="check_gt_detections",
    )((use_gt_detections, roi_input_non_ragged, detection_bboxes_non_ragged))

    # Reconstruct the ragged tensor.
    return tf.RaggedTensor.from_nested_row_splits(
        detection_geometry_values, detection_geometry_splits
    )


def build_combined_model(
    config: ModelConfig,
    encoder: Optional[tf.keras.Model] = None,
) -> tf.keras.Model:
    """
    Builds the combined detection + tracking model.

    Args:
        config: The model configuration.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.

    Returns:
        The model it created.

    """
    # Build the detector/feature extractor.
    image_feature_extractor, detector = build_detection_model(
        config, encoder=encoder
    )

    # Input for the current video frame.
    current_frames = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    # Input for the previous video frame.
    previous_frames = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.TRACKLETS_FRAME.value,
    )
    # Confidence threshold to use for filtering detections.
    confidence_threshold = layers.Input(
        shape=(),
        dtype=tf.float32,
        batch_size=1,
        name=ModelInputs.CONFIDENCE_THRESHOLD.value,
    )

    # Apply the detection model to the input frames.
    (
        detection_heatmap,
        detection_dense_geometry,
        detection_bboxes_with_conf,
    ) = _apply_detector(
        detector,
        frames=current_frames,
        confidence_threshold=confidence_threshold[0],
    )

    geometry_input_shape = (None, 4)
    # In all cases, we need to manually provide the tracklet bounding boxes.
    # These can either be the ground-truth, during training, or the
    # detections from the previous frame, during online inference.
    tracklet_geometry = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.TRACKLET_GEOMETRY.value,
    )

    # Indicates whether the model is in training mode, which impacts where we
    # get the tracking ROIs from.
    use_gt_detections = layers.Input(
        shape=(),
        dtype=tf.bool,
        batch_size=1,
        name=ModelInputs.USE_GT_DETECTIONS.value,
    )

    # Create the GT detection bounding box inputs, which are needed for
    # training.
    tracker_roi_input = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.DETECTION_GEOMETRY.value,
    )

    # Remove the confidence value from the detected bboxes, since we don't
    # need it for tracking.
    detection_bboxes = detection_bboxes_with_conf[:, :, :4]
    detection_geometry_for_tracker = _choose_rois_for_tracker(
        detection_bboxes=detection_bboxes,
        tracker_roi_input=tracker_roi_input,
        use_gt_detections=use_gt_detections,
    )

    # Extract the image features.
    current_frame_features = image_feature_extractor(current_frames)
    previous_frame_features = image_feature_extractor(previous_frames)

    # Extract the appearance features.
    detection_features = _extract_appearance_features(
        bbox_geometry=detection_geometry_for_tracker,
        image_features=current_frame_features,
        config=config,
    )
    tracklet_features = _extract_appearance_features(
        bbox_geometry=tracklet_geometry,
        image_features=previous_frame_features,
        config=config,
    )

    sinkhorn, hungarian = compute_association(
        detections_app_features=detection_features,
        tracklets_app_features=tracklet_features,
        detections_geometry=detection_geometry_for_tracker,
        tracklets_geometry=tracklet_geometry,
        config=config,
    )
    return tf.keras.Model(
        inputs=[
            current_frames,
            previous_frames,
            tracklet_geometry,
            tracker_roi_input,
            use_gt_detections,
            confidence_threshold,
        ],
        outputs=[
            detection_heatmap,
            detection_dense_geometry,
            detection_bboxes_with_conf,
            sinkhorn,
            hungarian,
        ],
        name="gcnnmatch_end_to_end",
    )
