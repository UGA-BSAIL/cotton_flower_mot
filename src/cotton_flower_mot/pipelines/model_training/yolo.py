"""
Utilities for implementing detection with a pre-trained YOLO model.
"""


from pathlib import Path
import tensorflow as tf
from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets

layers = tf.keras.layers


_RAW_MODEL_INPUT_SHAPE = (544, 960, 3)
"""
Input shape of the raw model.
"""


def _preprocess_inputs(images: tf.Tensor) -> tf.Tensor:
    """
    Preprocesses image inputs for YOLO.

    Args:
        images: The raw images, as a `uint8` tensor.

    Returns:
        The preprocessed images.

    """
    # Convert from BGR to RGB.
    images_rgb = images[:, :, :, ::-1]
    # Convert to floats.
    images_float = tf.cast(images_rgb, tf.float32)
    # Resize to the correct input size.
    images_float = tf.image.resize(images_float, _RAW_MODEL_INPUT_SHAPE[:2])
    return images_float / 255.0


def load_yolo(model_path: Path, *, config: ModelConfig) -> tf.keras.Model:
    """
    Loads a YOLO model, and adds the glue necessary to make it
    work with the tracking pipeline.

    Args:
        model_path: The path to the `saved model` directory.
        config: The model configuration to use.

    Returns:
        The loaded model.

    """
    # Load the model.
    raw_model = tf.keras.models.load_model(model_path)

    # Create a new model that's compatible with the pipeline.
    image_input = layers.Input(
        input_shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    images_preprocessed = layers.Lambda(_preprocess_inputs, name="preprocess")

    boxes, features = raw_model(images_preprocessed)

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
        inputs=[image_input], outputs=[boxes, features], name="yolo_detector"
    )
