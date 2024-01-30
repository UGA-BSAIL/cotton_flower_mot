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
