"""
Implementation of the CenterNet detector model.
"""


from typing import Optional, Tuple

import tensorflow as tf
from keras import layers
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Cropping2D,
    Dropout,
    ReLU,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.python.keras.regularizers import l2

from ..config import ModelConfig
from ..schemas import (
    ColorizationTargets,
    ModelInputs,
    ModelTargets,
    RotNetTargets,
)
from .layers import BnActConv, PeakLayer
from .layers.feature_extractors import efficientnet


def _build_decoder(
    multi_scale_features: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the decoder portion of the network. This is typically used on top
    of a pre-trained feature extractor in a transfer-learning setup.

    Args:
        multi_scale_features: Features extracted from different layers of the
            encoder. Each set of features is expected to be half the spatial
            resolution of the previous one.

    Returns:
        The decoder features.

    """
    scale1, scale2, scale3, scale4 = multi_scale_features

    scale4 = Dropout(rate=0.5)(scale4)
    scale3 = Dropout(rate=0.4)(scale3)
    scale2 = Dropout(rate=0.3)(scale2)
    scale1 = Dropout(rate=0.2)(scale1)
    x = scale4

    # decoder
    x = Conv2D(
        256,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # We need a little padding to make the layer sizes line up.
    x = ZeroPadding2D(((1, 0), (0, 0)))(x)
    x = Concatenate()([scale3, x])
    x = Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    scale3_merged = ReLU()(x)

    x = Conv2D(
        128,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(scale3_merged))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(((1, 0), (0, 0)))(x)
    x = Concatenate()([scale2, x])
    x = Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        64,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(((1, 0), (0, 0)))(x)
    x = Concatenate()([scale1, x])
    x = Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    return ReLU()(x), scale3_merged


def _build_prediction_head(
    features: tf.Tensor, *, output_channels: int, name: Optional[str] = None
) -> tf.Tensor:
    """
    Builds a prediction head for the model.

    Args:
        features: The features from the backbone.
        output_channels: The number of output channels we want.
        name: Name for the output layer.

    Returns:
        The prediction maps.

    """
    conv1_1 = BnActConv(256, 3, activation="relu", padding="same")(features)
    return BnActConv(output_channels, 1, padding="same", name=name)(conv1_1)


def compute_sparse_predictions(
    *, confidence_masks: tf.Tensor, sizes: tf.Tensor, offsets: tf.Tensor
) -> tf.RaggedTensor:
    """
    Computes sparse predictions given the estimated dense heatmap, sizes,
    and offsets. This is very useful for inference.

    Args:
        confidence_masks: The estimated center point masks, with a positive
            confidence score at the locations of objects and 0 everywhere else,
            with the shape `[batch, height, width, 1]`.
        sizes: The estimated size features, of the form
            `[batch, height, width, 2]`.
        offsets: The estimated offset features, of the form
            `[batch, height, width, 2]`.

    Returns:
        The computed bounding boxes, with the shape
        `[batch, num_boxes, 5]`, where the second dimension is ragged, and the
        third has the form `[center_x, center_y, width, height, confidence]`.

    """
    confidence_masks = tf.ensure_shape(confidence_masks, (None, None, None, 1))
    confidence_masks = confidence_masks[..., 0]
    confidence_masks = tf.cast(confidence_masks, tf.float32)
    center_masks = tf.greater(confidence_masks, 0.0)
    sizes = tf.ensure_shape(sizes, (None, None, None, 2))
    sizes = tf.cast(sizes, tf.float32)
    offsets = tf.ensure_shape(offsets, (None, None, None, 2))
    offsets = tf.cast(offsets, tf.float32)
    mask_shape = tf.shape(confidence_masks)[1:3]
    batch_size = tf.shape(confidence_masks)[0]

    # Mask the offsets and sizes.
    sparse_sizes = tf.boolean_mask(sizes, center_masks)
    sparse_offsets = tf.boolean_mask(offsets, center_masks)
    sparse_confidence = tf.boolean_mask(confidence_masks, center_masks)
    sparse_confidence = tf.expand_dims(sparse_confidence, 1)

    # Figure out the sparse center points.
    sparse_centers = tf.where(center_masks)
    center_points = sparse_centers[:, 1:]
    # Convert to normalized coordinates.
    center_points = tf.cast(center_points, tf.float32) / tf.cast(
        mask_shape, tf.float32
    )

    # Coerce all attributes into a ragged tensor by batch.
    batch_index = sparse_centers[:, 0]
    possible_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1)
    row_lengths = tf.reduce_sum(
        tf.cast(batch_index == possible_indices, tf.int64), axis=1
    )
    ragged_center_points = tf.RaggedTensor.from_row_lengths(
        center_points[..., ::-1], row_lengths
    )
    ragged_sizes = tf.RaggedTensor.from_row_lengths(sparse_sizes, row_lengths)
    ragged_offsets = tf.RaggedTensor.from_row_lengths(
        sparse_offsets, row_lengths
    )
    ragged_confidence = tf.RaggedTensor.from_row_lengths(
        sparse_confidence, row_lengths
    )

    # Nudge them by the offsets.
    ragged_center_points += ragged_offsets
    # Add the size.
    return tf.concat(
        (ragged_center_points, ragged_sizes, ragged_confidence), axis=2
    )


def _build_common(
    input_shape: Tuple[int, int, int],
    pretrained: bool = True,
    encoder: Optional[tf.keras.Model] = None,
) -> Tuple[tf.keras.Input, Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Builds the portions of the model that are common to all setups.

    Args:
        input_shape: The shape of the image input to the model.
        pretrained: Whether to initialize with pretrained `ImageNet` weights.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.

    Returns:
        The input layer, and the final extracted features.

    """
    images = layers.Input(
        shape=input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )

    def _normalize(_images: tf.Tensor) -> tf.Tensor:
        # Normalize the images before putting them through the model.
        return tf.cast(_images, tf.keras.backend.floatx())
        # return tf.image.per_image_standardization(float_images)
        # return tf.keras.applications.resnet_v2.preprocess_input(float_images)

    normalized = layers.Lambda(_normalize, name="normalize")(images)

    # Build the model.
    if encoder is not None:
        encoder_features = encoder(normalized)
    else:
        encoder_features = efficientnet(
            image_input=normalized,
            input_shape=input_shape,
            pretrained=pretrained,
        )

    return images, encoder_features


def build_detection_model(
    config: ModelConfig, encoder: Optional[tf.keras.Model] = None
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds the detection model.

    Args:
        config: The model configuration.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.

    Returns:
        The feature extractor model, which only outputs the features from
        the backbone before the detection heads, and the full model, which
        outputs complete bounding boxes.

    """
    images, encoder_features = _build_common(
        input_shape=config.detection_model_input_shape, encoder=encoder
    )

    decoder_features, tracking_features = _build_decoder(encoder_features)

    heatmap = _build_prediction_head(decoder_features, output_channels=1)
    sizes = _build_prediction_head(decoder_features, output_channels=2)
    offsets = _build_prediction_head(decoder_features, output_channels=2)

    # The loss expects sizes and offsets to be merged.
    geometry = layers.Concatenate(
        name=ModelTargets.GEOMETRY_DENSE_PRED.value, dtype=tf.float32
    )((sizes, offsets))
    # Since we treat heatmaps like a pixel-wise classification problem,
    # apply sigmoid activation.
    heatmap = layers.Activation(
        "sigmoid", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(heatmap)

    # Compute sparse bounding boxes for convenience.
    confidence_mask = PeakLayer(with_confidence=True)(heatmap)
    bounding_boxes = layers.Lambda(
        lambda f: compute_sparse_predictions(
            confidence_masks=f[0], sizes=f[1], offsets=f[2]
        ),
        dtype=tf.float32,
        name=ModelTargets.GEOMETRY_SPARSE_PRED.value,
    )((confidence_mask, sizes, offsets))

    feature_extractor = tf.keras.Model(
        inputs=images, outputs=[tracking_features]
    )
    detection_model = tf.keras.Model(
        inputs=images, outputs=[heatmap, geometry, bounding_boxes]
    )

    return feature_extractor, detection_model


def build_rotnet_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use for RotNet pre-training.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    images = layers.Input(
        shape=config.rot_net_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )

    _, _, _, features = convnext(
        image_input=images,
        input_shape=config.rot_net_input_shape,
        pretrained=False,
    )

    # Add the classification head.
    class_head = BnActConv(4, 1, activation="relu", padding="same")(features)
    average_pool = layers.GlobalAveragePooling2D()(class_head)
    rotation_class = layers.Activation(
        "softmax", dtype=tf.float32, name=RotNetTargets.ROTATION_CLASS.value
    )(average_pool)

    return tf.keras.Model(inputs=images, outputs=rotation_class)


def build_colorization_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use for RotNet pre-training.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    images, encoder_features = _build_common(
        input_shape=config.colorization_input_shape, pretrained=False
    )

    # Pad the encoder features correctly.
    scale1, scale2, scale3, scale4 = encoder_features
    scale1 = layers.ZeroPadding2D(((2, 1), (0, 0)))(scale1)
    scale2 = layers.ZeroPadding2D(((1, 1), (0, 0)))(scale2)
    scale3 = layers.ZeroPadding2D(((0, 1), (0, 0)))(scale3)

    decoder_features = _build_decoder((scale1, scale2, scale3, scale4))

    # Add the prediction head.
    hue = _build_prediction_head(
        decoder_features,
        output_channels=32,
    )
    chroma = _build_prediction_head(
        decoder_features,
        output_channels=32,
    )
    hue = layers.Activation(
        "linear", dtype=tf.float32, name=ColorizationTargets.HUE_HIST.value
    )(hue)
    chroma = layers.Activation(
        "linear", dtype=tf.float32, name=ColorizationTargets.CHROMA_HIST.value
    )(chroma)

    return tf.keras.Model(inputs=images, outputs=[hue, chroma])
