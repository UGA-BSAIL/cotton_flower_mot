"""
Handles the details of inference with the GCNNMatch tracker system.
"""


from typing import Any

import keras
import tensorflow as tf
from keras import layers
from loguru import logger

from ..config import ModelConfig
from ..model_training.models_common import (
    apply_detector,
    apply_tracker,
    make_tracking_inputs,
)
from ..model_training.ragged_utils import ragged_map_fn


def _filter_detections(
    detections: tf.RaggedTensor,
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

        nms_indices = tf.image.non_max_suppression(
            boxes,
            confidence,
            max_output_size=15,
            iou_threshold=nms_iou_threshold,
            score_threshold=confidence_threshold,
        )

        # Get the actual bounding boxes again.
        return tf.gather(detections_, nms_indices, axis=0)

    def _do_nms(detections_: tf.RaggedTensor) -> tf.RaggedTensor:
        return tf.map_fn(
            _single_frame_nms,
            detections_,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None, 5], dtype=tf.float32, ragged_rank=0
            ),
        )

    return layers.Lambda(_do_nms, name="detections_nms")(detections)


def build_inference_model(
    training_model: keras.Model, *, config: ModelConfig, **kwargs: Any
) -> keras.Model:
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
        **kwargs: Will be forwarded to `_filter_detections`.

    Returns:
        The model that it created.

    """
    logger.debug("Building inference model...")

    (
        current_frames_input,
        last_frames_input,
        tracklet_geometry_input,
        _,
    ) = make_tracking_inputs(config=config)

    # Extract the individual detection and tracking models.
    detector = training_model.get_layer("centernet_detector")
    tracker = training_model.get_layer("gcnnmatch")

    # Get the detections.
    heatmap, dense_geometry, detections = apply_detector(
        detector, frames=current_frames_input
    )
    detections = _filter_detections(detections, **kwargs)
    # Ignore the confidence for the detections.
    detections = detections[:, :, :4]

    # Apply the tracker using the detections.
    sinkhorn, assignment = apply_tracker(
        tracker,
        current_frames=current_frames_input,
        previous_frames=last_frames_input,
        tracklet_geometry=tracklet_geometry_input,
        detection_geometry=detections,
    )

    return keras.Model(
        inputs=[
            current_frames_input,
            last_frames_input,
            tracklet_geometry_input,
        ],
        outputs=[heatmap, dense_geometry, detections, sinkhorn, assignment],
        name="end_to_end_inference",
    )
