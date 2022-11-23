"""
Implements a combined detection + tracking model.
"""


from typing import Optional, Tuple

import keras
import tensorflow as tf
from keras import layers
from loguru import logger

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .centernet_model import build_detection_model
from .gcnn_model import build_tracking_model
from .models_common import make_tracking_inputs


def _apply_detector(
    detector: keras.Model,
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


def _apply_tracker(
    tracker: keras.Model,
    *,
    current_frames: tf.Tensor,
    previous_frames: tf.Tensor,
    tracklet_geometry: tf.RaggedTensor,
    detection_geometry: tf.RaggedTensor
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """
    Applies the tracker model to an input.

    Args:
        tracker: The tracker model.
        current_frames: The current input frames.
        previous_frames: The previous input frames.
        tracklet_geometry: The bounding boxes for the tracked objects.
        detection_geometry: The bounding boxes for the new detections.

    Returns:
        The sinkhorn and assignment matrices.

    """
    sinkhorn, assignment = tracker(
        {
            ModelInputs.DETECTIONS_FRAME.value: current_frames,
            ModelInputs.TRACKLETS_FRAME.value: previous_frames,
            ModelInputs.DETECTION_GEOMETRY.value: detection_geometry,
            ModelInputs.TRACKLET_GEOMETRY.value: tracklet_geometry,
        }
    )

    # Ensure that the resulting layers have the correct names when we set
    # them as outputs.
    sinkhorn = layers.Activation(
        "linear", name=ModelTargets.SINKHORN.value, dtype=tf.float32
    )(sinkhorn)
    assignment = layers.Activation(
        "linear", name=ModelTargets.ASSIGNMENT.value, dtype=tf.float32
    )(assignment)

    return sinkhorn, assignment


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


def build_separate_models(
    config: ModelConfig, encoder: Optional[tf.keras.Model] = None
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds compatible detection and tracking models.

    Args:
        config: The model configuration.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.

    Returns:
        The detection and tracking models.

    """
    logger.debug("Building detection model...")

    # Build the detector/feature extractor.
    image_feature_extractor, detector = build_detection_model(
        config, encoder=encoder
    )

    logger.debug("Building tracking model...")
    tracking_model = build_tracking_model(
        config=config, feature_extractor=image_feature_extractor
    )

    return detector, tracking_model


def build_combined_model(
    config: ModelConfig, *, detector: keras.Model, tracker: keras.Model
) -> tf.keras.Model:
    """
    Builds the combined detection + tracking model.

    Args:
        config: The model configuration.
        detector: The detection model.
        tracker: The tracking model.

    Returns:
        The combined model it created.

    """
    logger.debug("Building the combined model...")

    (
        current_frames_input,
        last_frames_input,
        tracklet_geometry_input,
        detection_geometry_input,
    ) = make_tracking_inputs(config)

    # Apply the detection model to the input frames.
    (
        detection_heatmap,
        detection_dense_geometry,
        detection_bboxes_with_conf,
    ) = _apply_detector(
        detector,
        frames=current_frames_input,
    )
    # Apply the tracking model.
    sinkhorn, assignment = _apply_tracker(
        tracker,
        current_frames=current_frames_input,
        previous_frames=last_frames_input,
        tracklet_geometry=tracklet_geometry_input,
        detection_geometry=detection_geometry_input,
    )

    return tf.keras.Model(
        inputs=[
            current_frames_input,
            last_frames_input,
            tracklet_geometry_input,
            detection_geometry_input,
        ],
        outputs=[
            detection_heatmap,
            detection_dense_geometry,
            detection_bboxes_with_conf,
            sinkhorn,
            assignment,
        ],
        name="gcnnmatch_end_to_end",
    )
