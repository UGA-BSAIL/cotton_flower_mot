"""
Custom pooling layers.
"""


from typing import Any, Dict, Tuple

import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers


class PeakLayer(layers.Layer):
    """
    A layer that detects local maxima in an input image. It outputs a mask
    where local maximum pixels are 1, and all other pixels are zero. A pixel
    is considered a peak when it is the maximum of all its neighbors in a 3x3
    neighborhood.
    """

    _DILATION_KERNEL = tf.expand_dims(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32),
        axis=2,
    )
    """
    Dilation kernel to use for peak finding.
    """

    def __init__(
        self, *args: Any, with_confidence: bool = False, **kwargs: Any
    ):
        """
        Args:
            *args:  Will be forwarded to superclass.
            with_confidence: If true, it will set the positive values in the
                resulting peak mask to the confidence scores for that peak,
                instead of merely one.
            **kwargs: Will be forwarded to superclass.
        """
        super().__init__(*args, **kwargs)

        self._with_confidence = with_confidence

        # Create the sub-layers.
        self._is_peak = layers.Lambda(self._find_peaks, name="peak_finding")

    def _find_peaks(self, image: tf.Tensor) -> tf.Tensor:
        """
        Finds true local peaks within an image. To be a peak, a pixel must
        be greater than all surrounding pixels.

        Args:
            image: The image to find peaks in.

        Returns:
            The peak mask, where peak pixels are set to 1, and all others are
            zero.

        """
        kernel = tf.cast(self._DILATION_KERNEL, image.dtype)

        # Custom dilation for finding peaks.
        dilated = tf.nn.dilation2d(
            image,
            kernel,
            [1, 1, 1, 1],
            "SAME",
            "NHWC",
            [1, 1, 1, 1],
            name="peak_finding_dilation",
        )
        mask = tf.greater(image, dilated)

        if self._with_confidence:
            # Get the actual confidence at those points.
            return image * tf.cast(mask, image.dtype)
        return mask

    def call(self, inputs: tf.Tensor, **_: Any) -> tf.Tensor:
        return self._is_peak(inputs)

    def get_config(self) -> Dict[str, Any]:
        return {"with_confidence": self._with_confidence, "name": self.name}


class RoiPooling(layers.Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun

    This code is largely copied from here:
    https://github.com/kbardool/keras-frcnn/blob/master/keras_frcnn/RoiPoolingConv.py

    Args:
        pool_size: Size of pooling region to use. pool_size = 7 will result
            in a 7x7 region.

    Input shape:
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img: `(batch_size, rows, cols, channels)`
        X_roi:
            `(batch_size,num_rois,4)` list of rois, with ordering (x,y,w,h),
                where the second dimension is ragged. These should be in
                "normalized" coordinates, i.e. with the values in [0, 1].
    Output shape:
        5D tensor with shape:
        `(batch_size, num_rois, pool_size, pool_size, channels)`, where the
        second dimension is ragged.
    """

    def __init__(self, pool_size: int, **kwargs: Any):
        self._pool_size = pool_size

        # Number of input channels. Will be set when the layer is built.
        self._num_input_channels = None
        # Batch size. Will be set when the layer it built.
        self._batch_size = None

        super().__init__(**kwargs)

    @staticmethod
    def _convert_to_two_corners(bboxes: tf.Tensor) -> tf.Tensor:
        """
        Converts bounding boxes from `[x, y, width, height]` format to
        `[min_y, min_x, max_y, max_x]`.

        Args:
            bboxes: The bounding boxes, with shape `[n_boxes, 4]`

        Returns:
            The converted boxes.

        """
        center = bboxes[:, :2]
        size = bboxes[:, 2:]

        min_corner = center - size / 2.0
        max_corner = center + size / 2.0

        # The x and y coordinates need to be reversed.
        return tf.concat([min_corner[:, ::-1], max_corner[:, ::-1]], axis=-1)

    def build(self, input_shape: Tuple[tf.TensorShape, tf.RaggedTensorSpec]):
        self._num_input_channels = input_shape[0][3]
        self._batch_size = input_shape[0][0]

    def compute_output_shape(self, _: Any):
        return (
            self._batch_size,
            None,
            self._pool_size,
            self._pool_size,
            self._num_input_channels,
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.RaggedTensor], **_: Any):
        assert len(inputs) == 2, "ROI pooling should have two inputs."
        images, rois = inputs

        # Determine which boxes are for which images.
        box_image_indices = rois.value_rowids()
        box_image_indices = tf.cast(box_image_indices, tf.int32)
        # ROIs need to be flat for use with cropping function.
        flat_rois = tf.cast(rois.merge_dims(0, 1), tf.float32)
        # They also need to be in the expected format.
        flat_rois = self._convert_to_two_corners(flat_rois)

        roi_crops = tf.image.crop_and_resize(
            images,
            boxes=flat_rois,
            box_indices=box_image_indices,
            crop_size=(self._pool_size, self._pool_size),
        )
        # Always nice to have the static shape...
        roi_crops = tf.ensure_shape(
            roi_crops,
            (
                None,
                self._pool_size,
                self._pool_size,
                self._num_input_channels,
            ),
        )

        # Separate the batch and ROI dimensions again, which will make our
        # final output ragged.
        return tf.RaggedTensor.from_value_rowids(
            roi_crops, box_image_indices, nrows=rois.nrows(out_type=tf.int32)
        )

    def get_config(self):
        config = {"pool_size": self._pool_size}
        base_config = super().get_config()
        return dict(**base_config, **config)
