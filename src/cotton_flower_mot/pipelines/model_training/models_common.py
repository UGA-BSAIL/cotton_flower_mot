from keras import layers
from typing import Tuple
import tensorflow as tf
from ..config import ModelConfig
from ..schemas import ModelInputs


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
