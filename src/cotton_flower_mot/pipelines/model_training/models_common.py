from keras import layers
from typing import Tuple
import tensorflow as tf
from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
import keras


def make_tracking_inputs(
    config: ModelConfig,
) -> Tuple[tf.Tensor, tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]:
    """
    Creates inputs that are used by all tracking models.

    Args:
        config: The model configuration.

    Returns:
        The current frame input, previous frame input, tracklet geometry input,
        and detection geometry input.

    """
    # Input for the current video frame.
    current_frames_input = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    # Input for the previous video frame.
    last_frames_input = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.TRACKLETS_FRAME.value,
    )

    geometry_input_shape = (None, 4)
    # In all cases, we need to manually provide the tracklet bounding boxes.
    # These can either be the ground-truth, during training, or the
    # detections from the previous frame, during online inference.
    tracklet_geometry_input = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.TRACKLET_GEOMETRY.value,
    )
    # Detection geometry can either be the ground-truth boxes (during training),
    # or the detected boxes (during inference).
    detection_geometry_input = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.DETECTION_GEOMETRY.value,
    )

    return (
        current_frames_input,
        last_frames_input,
        tracklet_geometry_input,
        detection_geometry_input,
    )


def apply_detector(
    detector: keras.Model,
    *,
    frames: tf.Tensor,
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


def apply_tracker(
    tracker: keras.Model,
    *,
    current_frames: tf.Tensor,
    previous_frames: tf.Tensor,
    tracklet_geometry: tf.RaggedTensor,
    detection_geometry: tf.RaggedTensor,
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
