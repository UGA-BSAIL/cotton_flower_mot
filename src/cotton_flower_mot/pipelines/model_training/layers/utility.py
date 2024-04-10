"""
Common utility layers used in various places.
"""

from typing import Any, Dict, Optional, Type, TypeVar

import tensorflow as tf
from tensorflow.keras import layers


class _BnActLayer(layers.Layer):
    """
    Base layer for composite layers that apply BN and pre-activation.
    """

    def __init__(
        self,
        *args: Any,
        layer: Type[layers.Layer],
        activation: Optional[str] = "relu",
        name: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Args:
            *args: Will be forwarded to the layer instance.
            layer: The type of layer to apply after pre-activation.
            activation: Activation to use.
            name: The name of this layer.
            **kwargs: Will be forwarded to the layer instance.
        """
        super().__init__(name=name)

        self._layer_args = args
        self._layer_kwargs = kwargs
        self._activation = activation

        # Initialize the sub-layers.
        self._layer = layer(*args, **kwargs)
        self._norm = layers.BatchNormalization()
        self._act = layers.Activation(activation)

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        return self._layer(self._act(self._norm(inputs, training=training)))

    def get_config(self) -> Dict[str, Any]:
        """
        Gets a partial configuration for this layer. Note that this isn't
        a complete configuration, because the layer type is not known.

        Returns:
            The partial configuration.

        """
        return dict(
            layer_args=self._layer_args,
            layer_kwargs=self._layer_kwargs,
            activation=self._activation,
            name=self.name,
        )

    SubType = TypeVar("SubType")

    @classmethod
    def from_config(cls: Type[SubType], config: Dict[str, Any]) -> SubType:
        """
        Creates a new layer from a partial configuration.

        Args:
            config: The partial configuration.

        Returns:
            The created instance.

        """
        layer_args = config.pop("layer_args")
        layer_kwargs = config.pop("layer_kwargs")
        return cls(*layer_args, **layer_kwargs, **config)


class BnActConv(_BnActLayer):
    """
    Small helper layer that applies batch normalization, ReLU, and convolution
    in that order.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, layer=layers.Conv2D, **kwargs)


class BnActDense(_BnActLayer):
    """
    Small helper layer that applies batch normalization, ReLU, and a dense
    layer in that order.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, layer=layers.Dense, **kwargs)
