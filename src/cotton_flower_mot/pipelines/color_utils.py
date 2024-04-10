"""
Utilities for converting color data.
"""

import tensorflow as tf


def rgb_to_hcl(image: tf.Tensor) -> tf.Tensor:
    """
    Converts an image from RGB to the hue, chroma, lightness space.

    Args:
        image: The image to convert. All pixels should be in [0, 1].

    Returns: The HCL image.

    """
    # Convert the image to floats.
    float_image = tf.cast(image, tf.float32)
    rescaled_image = float_image / 255.0

    hsv = tf.image.rgb_to_hsv(rescaled_image)

    # Convert HSV to HCL.
    hue = hsv[:, :, 0]
    chroma = hsv[:, :, 1] * hsv[:, :, 2]
    lightness = hsv[:, :, 2] - chroma / 2.0

    return tf.stack([hue, chroma, lightness], axis=-1)
