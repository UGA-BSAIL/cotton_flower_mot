"""
Defines custom losses.
"""


from typing import Any, Dict

import tensorflow as tf

from ..heat_maps import make_point_annotation_map, trim_out_of_bounds
from ..schemas import ModelTargets
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch
from .ragged_utils import ragged_map_fn


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Implementation of binary cross-entropy loss that allows us to weight
    positive and negative samples differently.
    """

    _EPSILON = tf.constant(0.0001)
    """
    Small constant value to avoid log(0).
    """

    @staticmethod
    def _flatten(tensor: MaybeRagged) -> MaybeRagged:
        """
        Flattens a tensor, leaving the batch dimension intact.

        Args:
            tensor: The tensor or `RaggedTensor` to flatten.

        Returns:
            The same tensor, in 2D.

        """
        num_dims = len(tensor.shape)
        if num_dims <= 2:
            # We're already flat.
            return tensor

        if type(tensor) == tf.RaggedTensor:
            return tensor.merge_dims(1, num_dims - 1)
        else:
            batch_size = tf.shape(tensor)[0]
            flat_shape = tf.stack([batch_size, -1])
            return tf.reshape(tensor, flat_shape)

    def call(self, y_true: MaybeRagged, y_pred: MaybeRagged) -> tf.Tensor:
        y_true, y_pred = correct_ragged_mismatch(y_true, y_pred)
        y_true = self._flatten(y_true)
        y_pred = self._flatten(y_pred)

        # Because of how Sinkhorn works, we could get some values > 1.
        y_true = tf.minimum(1.0, y_true)
        y_pred = tf.minimum(1.0, y_pred)

        # Calculate the fraction of samples that are positive.
        num_positive = tf.reduce_sum(y_true, axis=1)
        if type(y_true) == tf.RaggedTensor:
            num_total = y_true.row_lengths()
        else:
            num_total = tf.shape(y_true)[1]
        # Determine the weights.
        negative_weights = num_positive / tf.maximum(
            tf.cast(num_total, tf.float32), self._EPSILON
        )
        negative_weights = tf.maximum(negative_weights, self._EPSILON)
        # Make sure it broadcasts correctly.
        negative_weights = tf.expand_dims(negative_weights, axis=-1)

        positive_samples = y_true * tf.math.log(y_pred + self._EPSILON)
        one = tf.constant(1.0)
        negative_samples = (one - y_true) * tf.math.log(
            one - y_pred + self._EPSILON
        )

        weighted_losses = (
            positive_samples + negative_weights * negative_samples
        )
        num_samples = tf.cast(tf.size(positive_samples), tf.float32)
        num_samples = tf.stop_gradient(num_samples)
        return -tf.reduce_sum(weighted_losses) / num_samples


class HeatMapFocalLoss(tf.keras.losses.Loss):
    """
    Implements a penalty-reduced pixel-wise logistic regression with focal loss,
    as described in https://arxiv.org/pdf/1904.07850.pdf
    """

    _EPSILON = tf.constant(0.0001)
    """
    Small constant value to avoid log(0).
    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        positive_loss_weight: float = 1.0,
        **kwargs: Any,
    ):
        """
        Args:
            alpha: Alpha parameter for the focal loss.
            beta: Beta parameter for the focal loss.
            positive_loss_weight: Additional weight to give the positive
                component of the loss. This is to help balance the
                preponderance of negative samples.
            **kwargs: Will be forwarded to superclass constructor.
        """
        super().__init__(**kwargs)

        self._alpha = tf.constant(alpha)
        self._beta = tf.constant(beta)
        self._positive_loss_weight = tf.constant(positive_loss_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        one = tf.constant(1.0)

        # Loss we use at "positive" locations.
        positive_loss = tf.pow(one - y_pred, self._alpha) * tf.math.log(
            y_pred + self._EPSILON
        )
        positive_loss *= self._positive_loss_weight
        # Loss we use at "negative" locations.
        negative_loss = (
            tf.pow(one - y_true, self._beta)
            * tf.pow(y_pred, self._alpha)
            * tf.math.log(one - y_pred + self._EPSILON)
        )

        # Figure out which locations are positive and which are negative.
        positive_mask = tf.equal(y_true, 1.0)
        pixel_wise_loss = tf.where(positive_mask, positive_loss, negative_loss)

        mean_loss = -tf.reduce_sum(pixel_wise_loss)
        # Normalize by the number of keypoints.
        num_points = tf.experimental.numpy.count_nonzero(positive_mask)
        num_points = tf.stop_gradient(num_points)
        return tf.cond(
            num_points > 0,
            lambda: mean_loss / tf.cast(num_points, tf.float32),
            lambda: mean_loss,
        )


class GeometryL1Loss(tf.keras.losses.Loss):
    """
    A custom sparse L1 loss specifically designed to work on the "geometry"
    target of the detection model. This target is unique because the
    predictions come in the form of a dense feature map, while the
    ground-truth comes in the form of a sparse vector.
    """

    def __init__(
        self, *, size_weight: float, offset_weight: float, **kwargs: Any
    ):
        """
        Args:
            size_weight: The weight to use for the size loss.
            offset_weight: The weight to use for the offset loss.
            **kwargs: Will be forwarded to superclass constructor.

        """
        super().__init__(**kwargs)

        self._size_weight = tf.constant(size_weight)
        self._offset_weight = tf.constant(offset_weight)

    @staticmethod
    def _sparse_l1_loss(
        dense_predictions: tf.Tensor, sparse_truth: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes sparse L1 loss between dense predictions and sparse GT.

        Args:
            dense_predictions: The dense predictions. Should be a feature map
                with the shape `[height, width]`.
            sparse_truth: The sparse ground-truth. Should be a matrix with
                rows the form `[x, y, value]`. The x and y coordinates
                specify the point in the predicted feature map that we are
                providing supervision for. Any points that are not specified
                in the GT will not be included in the loss. The value is the
                GT value for that point.

        Returns:
            The computed loss.

        """
        sparse_truth = tf.ensure_shape(sparse_truth, (None, 3))

        # Trim out-of-bounds points manually since we subtract directly from
        # the ground-truth later.
        sparse_truth = trim_out_of_bounds(sparse_truth)

        # Create the point mask.
        center_points = sparse_truth[:, :2]
        # See https://github.com/tensorflow/tensorrt/issues/118 for
        # explanation of indexing.
        map_size = tf.shape(dense_predictions)[..., ::-1]

        # Create a dense ground-truth map.
        truth_values = sparse_truth[:, 2]
        dense_truth = make_point_annotation_map(
            center_points, map_size=map_size, point_values=truth_values
        )

        # Calculate the loss between the dense ground truth and dense
        # predictions.
        dense_l1 = tf.abs(dense_truth - dense_predictions)

        # We only actually care about the points where a real object is.
        point_mask = make_point_annotation_map(
            center_points, map_size=map_size
        )
        point_mask = tf.stop_gradient(point_mask)
        point_mask = tf.cast(point_mask, tf.bool)
        sparse_l1 = tf.boolean_mask(dense_l1, point_mask)

        # It's possible that we have no points here. In that case, the L1
        # loss should be zero.
        return tf.cond(
            tf.size(sparse_l1) > 0,
            lambda: tf.reduce_mean(sparse_l1),
            lambda: 0.0,
        )

    @classmethod
    def _batch_sparse_l1_loss(
        cls, dense_predictions: tf.Tensor, sparse_truth: tf.RaggedTensor
    ) -> tf.Tensor:
        """
        Similar to `sparse_l1_loss` except works on a batch of data.

        Args:
            dense_predictions: The dense predictions. Should be feature maps
                with the shape `[batch, height, width]`.
            sparse_truth: The sparse ground-truth. Should be a matrix with
                shape `[batch, num_detections, 3]`, with rows of form
                `[x, y, value]`. The x and y coordinates specify the point in
                the predicted feature map that we are providing supervision
                for. Any points that are not specified in the GT will not be
                included in the loss. The value is the GT value for that point.

        Returns:
            The computed loss.

        """
        element_losses = ragged_map_fn(
            lambda e: cls._sparse_l1_loss(e[0], e[1]),
            (dense_predictions, sparse_truth),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32),
        )

        return tf.reduce_mean(element_losses)

    def call(self, y_true: tf.RaggedTensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the loss given size and offset inputs.

        Args:
            y_true: The sparse ground-truth. Should be a matrix with
                shape `[batch, num_detections, 6]`, with rows of form
                `[center_x, center_y, width, height, offset_x, offset_y]`.
            y_pred: The dense predictions. Should be feature maps
                with the shape `[batch, height, width, 4]`, where the order
                of the final feature dimension is
                `[width, height, offset_x, offset_y]`.

        Returns:
            The computed loss for the sizes and offsets.

        """
        # Ground-truth needs to be prefixed with the point locations.
        center_points = y_true[..., :2]
        true_width = tf.concat((center_points, y_true[..., 2:3]), 2)
        true_height = tf.concat((center_points, y_true[..., 3:4]), 2)
        true_offset_x = tf.concat((center_points, y_true[..., 4:5]), 2)
        true_offset_y = tf.concat((center_points, y_true[..., 5:6]), 2)

        # Compute the loss for each element individually.
        width_loss = self._batch_sparse_l1_loss(y_pred[..., 0], true_width)
        height_loss = self._batch_sparse_l1_loss(y_pred[..., 1], true_height)
        offset_x_loss = self._batch_sparse_l1_loss(
            y_pred[..., 2], true_offset_x
        )
        offset_y_loss = self._batch_sparse_l1_loss(
            y_pred[..., 3], true_offset_y
        )

        return (width_loss + height_loss) * self._size_weight + (
            offset_x_loss + offset_y_loss
        ) * self._offset_weight


def make_losses(
    *,
    alpha: float,
    beta: float,
    size_weight: float,
    offset_weight: float,
) -> Dict[str, tf.keras.losses.Loss]:
    """
    Creates the losses to use for the model.

    Args:
        alpha: The alpha parameter to use for focal loss.
        beta: The beta parameter to use for focal loss.
        size_weight: The weight to use for the size loss.
        offset_weight: The weight to use for the offset loss.

    Returns:
        The losses that it created.

    """
    return {
        ModelTargets.SINKHORN.value: WeightedBinaryCrossEntropy(),
        # ModelTargets.HEATMAP.value: HeatMapFocalLoss(
        #     alpha=alpha,
        #     beta=beta,
        #     name="heatmap_loss",
        # ),
        # ModelTargets.GEOMETRY_DENSE_PRED.value: GeometryL1Loss(
        #     size_weight=size_weight,
        #     offset_weight=offset_weight,
        #     name="geometry_loss",
        # ),
    }
