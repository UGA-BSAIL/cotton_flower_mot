"""
Handles the details of inference with the GCNNMatch tracker system.
"""


from typing import Any, Tuple, Dict, Union

import keras
import tensorflow as tf
from keras import layers
from loguru import logger
from pathlib import Path

from ..config import ModelConfig
from ..model_training.models_common import (
    apply_detector,
    apply_tracker,
    make_tracking_inputs,
)
from ..model_training.layers import CUSTOM_LAYERS


def _filter_detections(
    detections: Union[tf.RaggedTensor, tf.Tensor],
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
) -> tf.RaggedTensor:
    """
    Filters detections before they're used in the tracker.

    Args:
        detections: The detections to filter. Should have shape
            `[None, (None), 5]`.
        confidence_threshold: Any detections with a lower confidence than
            this will be ignored.
        nms_iou_threshold: IOU threshold used for deciding whether boxes
            overlap too much when performing NMS.

    Returns:
        The filtered detections.

    """

    def _single_frame_nms(detections_: tf.Tensor) -> tf.Tensor:
        # Performs NMS on detections from a single frame.
        boxes = detections_[:, :4]
        confidence = detections_[:, 4]

        # Convert to (x1, y1, x2, y2).
        half_size = boxes[:, 2:] / 2
        boxes = tf.concat(
            [boxes[:, :2] - half_size, boxes[:, :2] + half_size], axis=1
        )

        nms_indices = tf.image.non_max_suppression(
            boxes,
            confidence,
            max_output_size=15,
            iou_threshold=nms_iou_threshold,
            score_threshold=confidence_threshold,
        )

        # Get the actual bounding boxes again.
        return tf.gather(detections_, nms_indices, axis=0)

    def _do_nms(
        detections_: Union[tf.RaggedTensor, tf.Tensor]
    ) -> tf.RaggedTensor:
        return tf.map_fn(
            _single_frame_nms,
            detections_,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None, 5], dtype=detections_.dtype, ragged_rank=0
            ),
        )

    return layers.Lambda(_do_nms, name="detections_nms")(detections)


def _config_to_float32(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a configuration dictionary to use float32 instead of mixed
    precision.

    Args:
        config: The configuration dictionary to convert.

    Returns:
        The converted dictionary.

    """
    config = config.copy()

    if "dtype" in config:
        # Set it directly.
        config["dtype"] = "float32"
    if "layers" in config:
        # Otherwise, it's nested.
        for child_layer in config["layers"]:
            child_layer["config"] = _config_to_float32(child_layer["config"])

    return config


def build_inference_model(
    training_model: keras.Model,
    *,
    config: ModelConfig,
    detector_model_path: Path,
    **kwargs: Any
) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs a specialized model that can be used for inference. This
    differs from the training model in that:
        - Detection bounding boxes are used for the tracking step instead of
          ground-truth bounding boxes.
        - Bounding boxes with low confidence scores are filtered out.
        - NMS is performed on the detections.

    Args:
        training_model: The training model with trained weights.
        config: The model configuration that was used for `training_model`.
        detector_model_path: The path to the pretrained detector model to use
            for inference.
        **kwargs: Will be forwarded to `_filter_detections`.

    Returns:
        - The tracking model, which takes appearance and geometry features as
          and outputs the sinkhorn and assignment matrices.
        - The detection model, which takes a frame as input and outputs
          bounding boxes and appearance features.

    """
    logger.debug("Building inference model...")

    # Re-create the model in case we saved in mixed precision and want
    # to infer with float32.
    def _clone_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        # Ensure that it uses float32.
        layer_config = _config_to_float32(layer.get_config())

        layer_copy = layer.__class__.from_config(layer_config)
        # Set the weights.
        layer_copy.set_weights(layer.get_weights())

        return layer_copy

    with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
        training_model = tf.keras.models.clone_model(
            training_model, clone_function=_clone_layer
        )

    (
        current_frames_input,
        last_frames_input,
        tracklet_geometry_input,
        detection_geometry_input,
    ) = make_tracking_inputs(config=config)

    # Extract the individual detection and tracking models.
    detector = training_model.get_layer("yolo_detector")
    appearance_extractor = training_model.get_layer("appearance_features")
    tracker = training_model.get_layer("gcnnmatch")

    # Use the inference version of the detection model.
    yolo_raw = detector.get_layer("pretrained_tf_1")
    yolo_raw.reload(detector_model_path)

    # Get the detections.
    detector_outputs = apply_detector(detector, frames=current_frames_input)
    image_features = detector_outputs[0]
    detections = detector_outputs[-1]
    detections = _filter_detections(detections, **kwargs)
    # Ignore the confidence for the detections.
    detections = tf.cast(detections[:, :, :4], tf.float32)

    # Extract appearance features for the detections.
    appearance_features = appearance_extractor(
        dict(image_features=image_features, geometry=detections),
    )

    detection_model = keras.Model(
        inputs=[current_frames_input],
        outputs=[detections, appearance_features],
        name="detector_inference",
    )
    return tracker, detection_model
