"""
Builds pre-trained feature extractors that are useful for transfer learning.
"""


from typing import Tuple

import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2S
from keras.applications.resnet_v2 import ResNet101V2
from .future.convnext import ConvNeXtBase
from keras import layers


def convnext(
    *,
    image_input: tf.Tensor,
    input_shape: Tuple[int, int, int],
    pretrained: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Creates a new ConvNeXt-based model.

    Args:
        image_input: The tensor containing (non-normalized) input images.
        input_shape: The shape of the inputs to the model.
        pretrained: If true, it will use pretrained `ImageNet` weights and
            fix those portions of the network. Otherwise, it will initialize
            randomly, and the whole thing will be trainable.

    Returns:
        The multiscale features extracted from the encoder, from the shallowest
        to deepest.

    """
    model = ConvNeXtBase(
        include_top=False,
        input_tensor=image_input,
        input_shape=input_shape,
        weights="imagenet" if pretrained else None,
    )
    model.trainable = not pretrained

    stage0 = model.get_layer(index=26).get_output_at(0)
    stage1 = model.get_layer(index=51).get_output_at(0)
    stage2 = model.get_layer(index=268).get_output_at(0)
    top = model.get_layer(index=294).get_output_at(0)

    return stage0, stage1, stage2, top


def efficientnet(
    *,
    image_input: tf.Tensor,
    input_shape: Tuple[int, int, int],
    pretrained: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Creates a new EfficientNet-based model.

    Args:
        image_input: The tensor containing (non-normalized) input images.
        input_shape: The shape of the inputs to the model.
        pretrained: If true, it will use pretrained `ImageNet` weights and
            fix those portions of the network. Otherwise, it will initialize
            randomly, and the whole thing will be trainable.

    Returns:
        The multiscale features extracted from the encoder, from the shallowest
        to deepest.

    """
    model = EfficientNetV2S(
        include_top=False,
        input_tensor=image_input,
        input_shape=input_shape,
        weights="imagenet" if pretrained else None,
    )
    model.trainable = not pretrained

    block2 = model.get_layer("block2d_add").get_output_at(0)
    block3 = model.get_layer("block3d_add").get_output_at(0)
    block5 = model.get_layer("block5i_add").get_output_at(0)
    top = model.get_layer("top_activation").get_output_at(0)

    # Fudge the sizes a little bit so everything's the same size as ConvNeXt.
    block3 = layers.Cropping2D(((0, 1), (0, 0)))(block3)
    block5 = layers.Cropping2D(((1, 0), (0, 0)))(block5)
    top = layers.Cropping2D(((0, 1), (0, 0)))(top)

    return block2, block3, block5, top


def resnet(
    *,
    image_input: tf.Tensor,
    input_shape: Tuple[int, int, int],
    pretrained: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Creates a new ResNet-based model.

    Args:
        image_input: The tensor containing (normalized) input images.
        input_shape: The shape of the inputs to the model.
        pretrained: If true, it will use pretrained `ImageNet` weights and
            fix those portions of the network. Otherwise, it will initialize
            randomly, and the whole thing will be trainable.

    Returns:
        The multiscale features extracted from the encoder, from the shallowest
        to deepest.

    """
    model = ResNet101V2(
        include_top=False,
        input_tensor=image_input,
        input_shape=input_shape,
        weights="imagenet" if pretrained else None,
    )
    model.trainable = not pretrained

    c2 = model.get_layer("conv2_block3_1_relu").get_output_at(0)
    c3 = model.get_layer("conv3_block4_1_relu").get_output_at(0)
    c4 = model.get_layer("conv4_block23_1_relu").get_output_at(0)
    c5 = model.get_layer("post_relu").get_output_at(0)

    return c2, c3, c4, c5
