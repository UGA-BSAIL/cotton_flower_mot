"""
Utilities for implementing detection with a pre-trained YOLO model.
"""


from functools import partial, update_wrapper
from pathlib import Path
from typing import Tuple

from loguru import logger

import tensorflow as tf
from ..config import ModelConfig
from ...schemas import ModelInputs, ModelTargets
from .layers.pretrained_tf import PretrainedTf

layers = tf.keras.layers


_RAW_MODEL_INPUT_SHAPE = (544, 960, 3)


def _preprocess_inputs(
    images: tf.Tensor, *, yolo_shape: Tuple[int, int, int]
) -> tf.Tensor:
    """
    Preprocesses image inputs for YOLO.

    Args:
        images: The raw images, as a `uint8` tensor.
        yolo_shape: The shape of the expected inputs to the YOLO model.

    Returns:
        The preprocessed images.

    """
    # Convert to floats.
    images_float = tf.cast(images, tf.float32)
    # Resize to the correct input size.
    images_float = tf.image.resize(images_float, yolo_shape[:2])
    return images_float / 255.0


def _postprocess_boxes(
    boxes: tf.Tensor, *, yolo_shape: Tuple[int, int, int]
) -> tf.Tensor:
    """
    Post-processes the bounding box outputs to make them more like what
    the other models produce.

    Args:
        boxes: The raw bounding box outputs.
        yolo_shape: The shape of the expected inputs to the YOLO model.

    Returns:
        The postprocessed boxes.

    """
    # Swap the bounding box dimensions here, because it generally expects
    # the box dimensions to come first, but YOLO puts them last.
    boxes = tf.transpose(boxes, perm=(0, 2, 1))
    # Also, box coordinates have to be normalized.
    return boxes / tf.constant(
        list(yolo_shape[:2][::-1]) * 2 + [1.0],
        dtype=boxes.dtype,
    )


def load_yolo(saved_model: Path, *, config: ModelConfig) -> tf.keras.Model:
    """
    Loads a YOLO model, and adds the glue necessary to make it
    work with the tracking pipeline.

    Args:
        saved_model: The path to the raw pretrained model.
        config: The model configuration to use.

    Returns:
        The loaded model.

    """
    logger.debug("Expected YOLO input shape: {}", config.raw_yolo_input_shape)
    preprocess = partial(
        _preprocess_inputs, yolo_shape=config.raw_yolo_input_shape
    )
    postprocess = partial(
        _postprocess_boxes, yolo_shape=config.raw_yolo_input_shape
    )
    preprocess = update_wrapper(preprocess, _preprocess_inputs)
    postprocess = update_wrapper(postprocess, _postprocess_boxes)

    # Create a new model that's compatible with the pipeline.
    image_input = layers.Input(
        shape=(None, None, 3),
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    images_preprocessed = layers.Lambda(preprocess, name="preprocess")(
        image_input
    )

    boxes, features = PretrainedTf(saved_model, name="yolo_raw")(
        images_preprocessed
    )
    boxes = layers.Lambda(postprocess, name="postprocess")(boxes)

    # Ensure the outputs have the right name and dtype.
    boxes = layers.Activation(
        "linear",
        name=ModelTargets.GEOMETRY_SPARSE_PRED.value,
        dtype=tf.float32,
    )(boxes)
    features = layers.Activation(
        "linear", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(features)

    return tf.keras.Model(
        inputs=[image_input], outputs=[features, boxes], name="yolo_detector"
    )
