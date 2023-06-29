from keras import layers
from typing import Tuple, Union
import tensorflow as tf
from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
import keras


def make_geometry_inputs() -> Tuple[layers.Input, layers.Input]:
    """
    Creates inputs for bounding box geometry.

    Returns:
        The tracklet and detection geometry inputs.

    """
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

    return tracklet_geometry_input, detection_geometry_input


def make_tracking_inputs(
    config: ModelConfig,
) -> Tuple[layers.Input, layers.Input, layers.Input, layers.Input]:
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

    tracklet_geometry_input, detection_geometry_input = make_geometry_inputs()

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
) -> Union[
    Tuple[tf.Tensor, tf.Tensor, tf.RaggedTensor],
    Tuple[tf.Tensor, tf.Tensor],
]:
    """
    Applies the detector model to an input.

    Args:
        detector: The detector model.
        frames: The input frames.

    Returns:
        The heatmaps, dense geometry predictions (if present), and bounding
        boxes.

    """
    raw_outputs = detector(frames)
    dense_geometry = None
    if len(raw_outputs) == 3:
        heatmap, dense_geometry, bboxes = raw_outputs

        dense_geometry = layers.Activation(
            "linear",
            name=ModelTargets.GEOMETRY_DENSE_PRED.value,
            dtype=tf.float32,
        )(dense_geometry)
    else:
        heatmap, bboxes = raw_outputs

    # Ensure that the resulting layers have the correct names when we set
    # them as outputs.
    heatmap = layers.Activation(
        "linear", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(heatmap)
    bboxes = layers.Activation(
        "linear",
        name=ModelTargets.GEOMETRY_SPARSE_PRED.value,
        dtype=tf.float32,
    )(bboxes)

    if dense_geometry is not None:
        return heatmap, dense_geometry, bboxes
    else:
        return heatmap, bboxes


def apply_tracker(
    tracker: keras.Model,
    *,
    tracklet_appearance: tf.RaggedTensor,
    detection_appearance: tf.RaggedTensor,
    tracklet_geometry: tf.RaggedTensor,
    detection_geometry: tf.RaggedTensor,
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """
    Applies the tracker model to an input.

    Args:
        tracker: The tracker model.
        tracklet_appearance: The appearance features for the tracked objects.
        detection_appearance: The appearance features for the new detections.
        tracklet_geometry: The bounding boxes for the tracked objects.
        detection_geometry: The bounding boxes for the new detections.

    Returns:
        The sinkhorn and assignment matrices.

    """
    sinkhorn, assignment = tracker(
        {
            ModelInputs.TRACKLET_APPEARANCE.value: tracklet_appearance,
            ModelInputs.DETECTION_APPEARANCE.value: detection_appearance,
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
