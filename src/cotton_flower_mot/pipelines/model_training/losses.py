"""
Defines custom losses.
"""


from typing import Any, Dict

import tensorflow as tf

from ..dataset_io import RotNetFeatureName
from ..heat_maps import make_point_annotation_map, trim_out_of_bounds
from ..schemas import ModelTargets
from .graph_utils import compute_pairwise_similarities
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch
from .ragged_utils import ragged_map_fn
from .similarity_utils import (
    aspect_ratio_penalty,
    compute_ious,
    distance_penalty,
)


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

        # Calculate the fraction of samples that are positive.
        num_positive = tf.reduce_sum(y_true, axis=1)
        if type(y_true) == tf.RaggedTensor:
            num_total = y_true.row_lengths()
        else:
            num_total = tf.shape(y_true)[1]
        # Determine the weights.
        negative_weights = num_positive / tf.cast(num_total, tf.float32)
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
        point_mask = make_point_annotation_map(
            center_points, map_size=map_size
        )
        point_mask = tf.cast(point_mask, tf.bool)

        # Extract the relevant point values with the mask.
        sparse_predictions = tf.boolean_mask(dense_predictions, point_mask)

        # We need to make sure the sparse truth is ordered the same way,
        # which is why we also represent it in dense form.
        truth_values = sparse_truth[:, 2]
        dense_truth = make_point_annotation_map(
            center_points, map_size=map_size, point_values=truth_values
        )
        ordered_sparse_truth = tf.boolean_mask(dense_truth, point_mask)

        # We should now be able to compare directly to the ground-truth.
        return tf.norm(ordered_sparse_truth - sparse_predictions, ord=1)

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
        # true_offset_x = tf.concat((center_points, y_true[..., 4:5]), 2)
        # true_offset_y = tf.concat((center_points, y_true[..., 5:6]), 2)

        # Compute the loss for each element individually.
        width_loss = self._batch_sparse_l1_loss(y_pred[..., 0], true_width)
        height_loss = self._batch_sparse_l1_loss(y_pred[..., 1], true_height)
        # offset_x_loss = self._batch_sparse_l1_loss(
        #     y_pred[..., 2], true_offset_x
        # )
        # offset_y_loss = self._batch_sparse_l1_loss(
        #     y_pred[..., 3], true_offset_y
        # )

        # return (width_loss + height_loss) * self._size_weight + (
        #     offset_x_loss + offset_y_loss
        # ) * self._offset_weight
        return (width_loss + height_loss) * self._size_weight


class CIOULoss(tf.keras.losses.Loss):
    """
    Implements cIOU loss.
    """

    def __init__(
        self,
        iou_weight: float = 1.0,
        distance_weight: float = 1.0,
        aspect_ratio_weight: float = 1.0,
        classification_weight: float = 1.0,
        positive_threshold: float = 0.35,
        negative_threshold: float = 0.75,
        **kwargs: Any,
    ):
        """
        Args:
            iou_weight: Weight to apply to the IOU component of the loss.
            distance_weight: Weight to apply to the center distance component
                of the loss.
            aspect_ratio_weight: Weight to apply to the aspect ratio component
                of the loss.
            positive_threshold: cIOU loss threshold below which we consider
                a box proposal to be positive.
            negative_threshold: cIOU loss threshold above which we consider a
                box proposal to be negative.
            classification_weight: How much we weight the portion of the loss
                that classifies positive vs. negative bounding boxes.
            **kwargs: Will be forwarded to superclass.

        """
        super().__init__(**kwargs)

        self._iou_weight = tf.constant(iou_weight)
        self._distance_weight = tf.constant(distance_weight)
        self._aspect_ratio_weight = tf.constant(aspect_ratio_weight)
        self._classification_weight = tf.constant(classification_weight)
        self._positive_threshold = tf.constant(positive_threshold)
        self._negative_threshold = tf.constant(negative_threshold)

    def _compute_loss(
        self, left_boxes: tf.Tensor, right_boxes: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the complete losses between two sets of paired bounding boxes.

        Args:
            left_boxes: The first set of boxes. Should have shape
                `[num_boxes, 4]`, where each row has the form
                `[center_x, center_y, size_x, size_y]`.
            right_boxes: The second set of boxes, in the same form.

        Returns:
            The calculated cIOU loss between each pair of boxes.

        """
        ious = compute_ious(left_boxes, right_boxes)
        distance = distance_penalty(left_boxes, right_boxes)
        aspect_ratio = aspect_ratio_penalty(left_boxes, right_boxes)

        return (
            self._iou_weight * (tf.constant(1.0) - ious)
            + self._distance_weight * distance
            + self._aspect_ratio_weight * aspect_ratio
        )

    @staticmethod
    def _compute_regression_loss(pairwise_losses: tf.Tensor) -> tf.Tensor:
        """
        Computes the bounding-box regression component of the loss.

        Args:
            pairwise_losses: The complete pairwise losses, with shape
                `[num_truth, num_predictions]`.

        Returns:
            The computed total bounding box regression loss for positive
            examples.

        """
        # Find the best predictions for each ground-truth box.
        truth_min_loss = tf.reduce_min(pairwise_losses, axis=1)

        # Stop it from going infinite if there are no data points.
        mean_loss = tf.reduce_mean(truth_min_loss)
        return tf.cond(
            tf.math.is_inf(mean_loss), lambda: 10.0, lambda: mean_loss
        )

    def _compute_classification_loss(
        self, pairwise_losses: tf.Tensor, confidence: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the total positive/negative classification component of the
        loss.

        Args:
            pairwise_losses: The complete pairwise losses, with shape
                `[num_truth, num_predictions]`.
            confidence: The confidence vector, containing corresponding
                confidences for each prediction.

        Returns:
            The total classification loss.

        """
        # Separate predictions into positive and negative examples.
        prediction_min_loss = tf.reduce_min(pairwise_losses, axis=0)
        positive_mask = tf.less_equal(
            prediction_min_loss, self._positive_threshold
        )
        negative_mask = tf.greater_equal(
            prediction_min_loss, self._negative_threshold
        )
        positive_confidences = tf.boolean_mask(confidence, positive_mask)
        negative_confidences = tf.boolean_mask(confidence, negative_mask)

        predictions = tf.concat(
            (positive_confidences, negative_confidences), 0
        )
        truth = tf.concat(
            (
                tf.ones_like(positive_confidences),
                tf.zeros_like(negative_confidences),
            ),
            0,
        )
        loss = tf.keras.losses.binary_crossentropy(truth, predictions)

        # Remove NaN values from loss.
        loss = tf.where(tf.math.is_nan(loss), 0.0, loss)
        return tf.reduce_mean(loss)

    def _loss_for_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the total loss for a single training example.

        Args:
            y_true: The true bounding boxes for the detections, with the shape
                `[num_detections, 4]`, where each row has the same layout as
                the input to `_compute_loss`.
            y_pred: The predicted bounding boxes for the detections, with the
                shape `[num_detections, 5]`, where each row has the same
                layout as the input to `_compute_loss` with the addition of a
                confidence column.

        Returns:
            The total loss for all bounding boxes in the image.

        """
        y_true = tf.ensure_shape(y_true, (None, 6))
        y_pred = tf.ensure_shape(y_pred, (None, 5))
        # Handle the confidence separately.
        confidence = y_pred[:, 4]
        y_pred = y_pred[:, :4]
        # We don't need the offsets.
        y_true = y_true[:, :4]

        # Keep only the top-100 predictions.
        confidence_order = tf.argsort(confidence, direction="DESCENDING")
        confidence = tf.gather(confidence, confidence_order[:100])
        y_pred = tf.gather(y_pred, confidence_order[:100])

        # Calculate the cIOU loss between every possible combination of boxes.
        losses = compute_pairwise_similarities(
            self._compute_loss,
            # It expects there to be a batch dimension.
            left_features=tf.expand_dims(y_true, 0),
            right_features=tf.expand_dims(y_pred, 0),
        )[0]

        regression_loss = self._compute_regression_loss(losses)
        class_loss = self._compute_classification_loss(losses, confidence)
        return regression_loss + self._classification_weight * class_loss

    def call(
        self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor
    ) -> tf.Tensor:
        """
        Args:
            y_true: The true sparse bounding box locations, with a shape of
                `[batch, num_detections, 6]`. The last dimension is a vector
                of the form
                `[center_x, center_y, size_x, size_y, offset_x, offset_y]`.
            y_pred: The predicted sparse bounding box locations, with a shape
                of `[batch, num_detections, 5]`. The last dimension is a
                vector of the form
                `[center_x, center_y, size_x, size_y, confidence]`.

        """
        tf.print("y_true shape:", y_true.bounding_shape())
        tf.print("y_pred shape:", y_pred.bounding_shape())
        image_losses = ragged_map_fn(
            lambda e: self._loss_for_image(e[0], e[1]),
            (y_true, y_pred),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32),
        )

        # Average across all examples.
        return tf.reduce_mean(image_losses)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            iou_weight=self._iou_weight.numpy().tolist(),
            distance_weight=self._distance_weight.numpy().tolist(),
            aspect_ratio_weight=self._aspect_ratio_weight.numpy().tolist(),
            classification_weight=self._classification_weight.numpy().tolist(),
            positive_threshold=self._positive_threshold.numpy().tolist(),
            name=self.name,
        )


def make_losses(
    *,
    alpha: float,
    beta: float,
    size_weight: float,
    offset_weight: float,
    positive_loss_weight: float,
) -> Dict[str, tf.keras.losses.Loss]:
    """
    Creates the losses to use for the model.

    Args:
        alpha: The alpha parameter to use for focal loss.
        beta: The beta parameter to use for focal loss.
        size_weight: The weight to use for the size loss.
        offset_weight: The weight to use for the offset loss.
        positive_loss_weight: Weight to give positive examples in heatmap loss.

    Returns:
        The losses that it created.

    """
    return {
        # ModelTargets.SINKHORN.value: WeightedBinaryCrossEntropy(),
        ModelTargets.HEATMAP.value: HeatMapFocalLoss(
            alpha=alpha,
            beta=beta,
            positive_loss_weight=positive_loss_weight,
            name="heatmap_loss",
        ),
        ModelTargets.GEOMETRY_DENSE_PRED.value: GeometryL1Loss(
            size_weight=size_weight,
            offset_weight=offset_weight,
            name="geometry_loss",
        ),
        ModelTargets.GEOMETRY_SPARSE_PRED.value: CIOULoss(
            classification_weight=0.1, name="ciou_loss"
        ),
    }


def make_rotnet_loss() -> Dict[str, tf.keras.losses.Loss]:
    """
    Creates the loss for the RotNet model.

    Returns:
        The loss for the RotNet model.

    """
    return {
        RotNetFeatureName.PREDICTIONS.value: tf.keras.losses.SparseCategoricalCrossentropy()
    }
