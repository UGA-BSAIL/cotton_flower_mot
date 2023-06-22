import tensorflow as tf
from pathlib import Path
from typing import Tuple, Any, Dict
from loguru import logger


layers = tf.keras.layers


class PretrainedTf(layers.Layer):
    """
    Represents a pre-trained TF model that we can use directly as a layer.
    """

    def __init__(self, model_path: Path, *args: Any, **kwargs: Any):
        """
        Args:
            model_path: Path to the TF model.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.
        """
        self._model_path = None
        self._tf_model = None

        super().__init__(*args, **kwargs)
        self.reload(model_path)

    def reload(self, model_path: Path) -> None:
        """
        Reloads the model.

        Args:
            model_path: The path to the new model.

        """
        logger.info("Loading TF model from {}.", model_path)
        self._model_path = model_path
        self._tf_model = tf.saved_model.load(model_path)

    def call(self, inputs: Tuple[tf.Tensor], *args: Any, **kwargs: Any):
        return self._tf_model(tf.cast(inputs, tf.float32))

    def get_config(self) -> Dict[str, Any]:
        return {"model_path": self._model_path}
