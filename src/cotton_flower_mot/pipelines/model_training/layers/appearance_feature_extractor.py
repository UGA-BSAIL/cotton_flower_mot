from typing import Any, Tuple, Dict

import numpy as np
import tensorflow as tf
from keras import layers

from . import BnActConv, RoiPooling


class AppearanceFeatureExtractor(layers.Layer):
    """
    Custom layer for extracting appearance features.
    """

    def __init__(self, *args: Any, roi_pooling_size: int, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the superclass.
            roi_pooling_size: The size to use for ROI pooling.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, **kwargs)
        self._roi_pooling_size = roi_pooling_size

        self._dropout = layers.Dropout(0.5)
        self._conv_1 = BnActConv(128, 3, padding="same", name="app_conv_1")
        self._conv_2 = BnActConv(256, 1, padding="same", name="app_conv_2")
        self._conv_3 = BnActConv(256, 1, padding="same", name="app_conv_3")

        self._res_conv = BnActConv(256, 1, padding="same", name="app_res")
        self._res_add = layers.Add()

        self._conv_squeeze = BnActConv(
            8, 1, activation="relu", name="app_squeeze"
        )
        self._roi_pooling = RoiPooling(roi_pooling_size)

        # Coerce the features to the correct shape.
        def _flatten_features(_features: tf.RaggedTensor) -> tf.RaggedTensor:
            # We have to go through this annoying process to ensure that the
            # static shape remains correct.
            inner_shape = _features.shape[-3:]
            num_flat_features = np.prod(inner_shape)
            flat_features = tf.reshape(
                _features.values,
                (-1, num_flat_features),
                name="appearance_flatten",
            )
            # flat_features = bound_numerics(flat_features)
            return _features.with_values(flat_features)

        self._flatten = layers.Lambda(_flatten_features)

    def call(
        self, inputs: Tuple[tf.RaggedTensor, tf.Tensor], *_, **__
    ) -> tf.RaggedTensor:
        """
        Args:
            inputs:
                bbox geometry: The bounding box information. Should have
                    shape `[batch_size, num_boxes, 4]`, where the second dimension is
                    ragged, and the third is ordered `[x, y, width, height]`.
                    tracklets.
                image features: The raw image features from the detector.
            *_:
            **__:

        Returns:
            The extracted appearance features. They are a `RaggedTensor`
            with the shape `[batch_size, n_nodes, n_features]`, where the second
            dimension is ragged.

        """
        bbox_geometry, image_features = inputs
        image_features_res = image_features

        image_features = self._dropout(image_features)
        image_features = self._conv_1(image_features)
        image_features = self._conv_2(image_features)
        image_features = self._conv_3(image_features)

        image_features_res = self._res_conv(image_features_res)
        image_features = self._res_add((image_features_res, image_features))

        image_features = self._conv_squeeze(image_features)
        feature_crops = self._roi_pooling((image_features, bbox_geometry))

        return self._flatten(feature_crops)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(dict(roi_pooling_size=self._roi_pooling_size))
        return config
