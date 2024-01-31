"""
Utilities for TFRT models.
"""
from pathlib import Path
from typing import Callable, Dict, Tuple, Any

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants

GraphFunc = Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
"""
Type alias for a function that runs the TF graph.
"""


def set_gpu_memory_limit(limit_mb: int) -> None:
    """
    Sets the GPU memory limit in TensorFlow. Note that this generally has to
    be done right after TF is first imported.

    Args:
        limit_mb: The memory limit to set.

    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Inference should not need much memory.
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
        )


def get_func_from_saved_model(saved_model_dir: Path) -> Tuple[GraphFunc, Any]:
    """
    Generates a graph function from a saved TFRT model.

    Args:
        saved_model_dir: The saved model directory.

    Returns:
        The graph function, and the saved model.

    """
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING]
    )
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]

    return graph_func, saved_model_loaded
