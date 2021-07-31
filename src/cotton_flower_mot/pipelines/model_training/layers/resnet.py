import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Cropping2D,
    Dropout,
    ReLU,
    UpSampling2D,
)
from tensorflow.keras.regularizers import l2

from ...config import ModelConfig


def resnet(
    *,
    image_input: tf.Tensor,
    config: ModelConfig,
):
    resnet = ResNet50V2(
        include_top=False,
        input_tensor=image_input,
        input_shape=config.detection_model_input_shape,
    )
    resnet.trainable = False

    # C5 (b, 16, 16, 512)
    c2 = resnet.get_layer("conv2_block3_1_relu").get_output_at(0)
    c3 = resnet.get_layer("conv3_block4_1_relu").get_output_at(0)
    c4 = resnet.get_layer("conv4_block6_1_relu").get_output_at(0)
    c5 = resnet.get_layer("conv5_block3_2_relu").get_output_at(0)

    c5 = Dropout(rate=0.5)(c5)
    c4 = Dropout(rate=0.4)(c4)
    c3 = Dropout(rate=0.3)(c3)
    c2 = Dropout(rate=0.2)(c2)
    x = c5

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
    x = Concatenate()([c4, x])
    x = Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    # (b, 32, 32, 512)
    x = ReLU()(x)

    x = Conv2D(
        128,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([c3, x])
    x = Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    # (b, 64, 64, 128)
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
    # Because of the way the resnet layer sizes shake out, we need to add
    # some slight cropping here.
    x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
    x = Concatenate()([c2, x])
    x = Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    # (b, 128, 128, 512)
    return ReLU()(x)
