"""
Custom metrics for the model.
"""


import abc
from typing import Any, Dict, Optional, cast

import tensorflow as tf

from ..schemas import ModelTargets
from .graph_utils import compute_pairwise_similarities
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch
from .ragged_utils import ragged_map_fn
from .similarity_utils import compute_ious


def count_id_switches(
    true_assignment: tf.RaggedTensor, pred_assignment: tf.RaggedTensor
) -> tf.Tensor:
    """
    Calculates the number of ID switches for a batch of image pairs.

    Args:
        true_assignment: The true hard assignment matrices, of shape
            `[batch_size, n_tracklets * n_detections]`.
        pred_assignment: The predicted hard assignment matrices, of shape
            `[batch_size, n_tracklets * n_detections]`.

    Returns:
        The number of ID switches for each item in the batch.

    """
    tf.print("true_assignment:", true_assignment[0], summarize=-1)
    tf.print("pred_assignment:", pred_assignment[0], summarize=-1)
    switched = tf.math.logical_xor(true_assignment, pred_assignment)
    switched = tf.cast(switched, tf.int32)
    # Division by 2 is because every time the assignment changes, it results
    # in one incorrect True value and one incorrect False value.
    return tf.reduce_sum(switched, axis=1) // 2


class IdSwitches(tf.keras.metrics.Metric):
    """
    A metric that tracks the number of ID switches.
    """

    def __init__(self, name="count_id_switches", **kwargs: Any):
        """
        Args:
            name: The name of this metric.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._num_id_switches = self.add_weight(
            name="num_switches", initializer="zeros", dtype=tf.int32
        )

    def update_state(
        self,
        y_true: MaybeRagged,
        y_pred: MaybeRagged,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_true, y_pred = correct_ragged_mismatch(y_true, y_pred)

        _id_switches = count_id_switches(y_true, y_pred)

        if sample_weight is not None:
            # Weight each sample.
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, _id_switches.shape)
            _id_switches *= sample_weight

        self._num_id_switches.assign_add(tf.reduce_sum(_id_switches))

    def result(self) -> tf.Tensor:
        return self._num_id_switches


def _compute_bbox_matches(
    y_true: tf.Tensor, y_pred: tf.Tensor, *, iou_threshold: float
) -> tf.Tensor:
    """
    Matches each predicted box to a ground-truth box based upon IOU.
    Args:
        y_true: The true bounding boxes for the detections, with the shape
                `[num_detections, 6]`, where each row has the same layout as
                the input to `update_state`.
        y_pred: The predicted bounding boxes for the detections, with the
                shape `[num_detections, 5]`, where each row has the same
                layout as the input to `update_state`.
        iou_threshold: The IOU threshold for considering two boxes to match.

    Returns:
        A tensor of shape [num_true, num_pred], where the rows represent
        ground-truth boxes, and the columns represent predicted boxes. A
        value of zero means that the two boxes do not match. Otherwise,
        this value is the confidence of the predicted box.

    """
    y_true = tf.ensure_shape(y_true, (None, 6))
    y_pred = tf.ensure_shape(y_pred, (None, 5))
    confidence = y_pred[:, 4]
    y_pred = y_pred[:, :4]
    y_true = y_true[:, :4]

    # Determine which predictions match up with the truth through IOU.
    ious = compute_pairwise_similarities(
        compute_ious,
        # It expects there to be a batch dimension.
        left_features=tf.expand_dims(y_true, 0),
        right_features=tf.expand_dims(y_pred, 0),
    )[0]
    # Anything below the threshold doesn't match.
    iou_matches = tf.greater_equal(ious, iou_threshold)

    matches_with_confidence = tf.cast(iou_matches, tf.float32) * confidence

    return matches_with_confidence


class _BboxMetric(tf.keras.metrics.Metric, abc.ABC):
    """
    A base class for metrics that operate on bounding boxes.
    """

    @abc.abstractmethod
    def _update_state_from_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> None:
        """
        Performs the state update with boxes from a single image.

        Args:
            y_true: The true bounding boxes for the detections, with the shape
                    `[num_detections, 6]`, where each row has the same layout as
                    the input to `update_state`.
            y_pred: The predicted bounding boxes for the detections, with the
                shape `[num_detections, 5]`, where each row has the same
                layout as the input to `update_state`.

        """

    def _update_state_from_image_with_return(
        self, *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        """
        Same as `_update_state_from_image`, except it returns a dummy value
        so we can use it with `map_fn`.

        Args:
            *args: Will be forwarded to `_update_state_from_image`.
            **kwargs: Will be forwarded to `_update_state_from_image`.

        Returns:
            Always returns 0.

        """
        self._update_state_from_image(*args, **kwargs)

        return tf.constant(0)

    def update_state(
        self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor, **_
    ) -> None:
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
        # Compute the metrics.
        ragged_map_fn(
            lambda e: self._update_state_from_image_with_return(e[0], e[1]),
            (y_true, y_pred),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int32),
        )


class AveragePrecision(_BboxMetric):
    """
    Calculates the average precision for an object detector.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        use_top_predictions: Optional[int] = 10,
        name: str = "average_precision",
        **kwargs: Any
    ):
        """
        Args:
            iou_threshold: The IOU threshold for considering a detection to
                be a true positive.
            name: The name of the metric.
            use_top_predictions: The number of top predictions to consider when
                computing the AP. If None, it will consider all of them.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._iou_threshold = tf.constant(iou_threshold)
        self._use_top_predictions = use_top_predictions

        # Keep track of performance metrics.
        self._auc = tf.keras.metrics.AUC(curve="PR")

    @staticmethod
    def _take_top_predictions(
        predictions: tf.Tensor, num_to_take: int
    ) -> tf.Tensor:
        """
        Takes only the N predictions with the highest confidence.

        Args:
            predictions: The predictions to filter, in the same format as
                `_update_auc_from_image`.
            num_to_take: The number of top predictions to take.

        Returns:
            The filtered predictions.

        """
        confidence = predictions[:, 4]

        confidence_order = tf.argsort(confidence, direction="DESCENDING")
        return tf.gather(predictions, confidence_order[:num_to_take])

    @staticmethod
    def _safe_argmax(
        values: tf.Tensor,
        axis: int,
        output_type: tf.dtypes.int64,
        **kwargs: Any
    ) -> tf.Tensor:
        """
        A version of `argmax` that safely handles the case where the
        reduction axis has a size of zero. In this case, it just returns an
        empty tensor.

        Args:
            values: The values to take the argmax of.
            axis: The axis to take the maximum on.
            output_type: Output data type.
            **kwargs: Will be forwarded to `tf.argmax`.

        Returns:
            The argmax result.

        """
        axis_size = tf.shape(values)[axis]
        return tf.cond(
            axis_size == 0,
            lambda: tf.constant([], dtype=output_type),
            lambda: tf.argmax(
                values, axis=axis, output_type=output_type, **kwargs
            ),
        )

    def _update_state_from_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> None:
        if self._use_top_predictions is not None:
            # Filter to top predictions.
            y_pred = self._take_top_predictions(
                y_pred, self._use_top_predictions
            )
        matches_with_confidence = _compute_bbox_matches(
            y_true, y_pred, iou_threshold=self._iou_threshold
        )

        # Confidence cannot realistically be less than 0, even if there are
        # no predictions.
        true_positive_confidence = tf.maximum(
            tf.reduce_max(matches_with_confidence, axis=1), 0.0
        )

        # Figure out which predictions are false-positives, and create a mask
        # for them.
        true_positive_indices = self._safe_argmax(
            matches_with_confidence, axis=1, output_type=tf.int32
        )
        true_positive_mask = tf.scatter_nd(
            tf.expand_dims(true_positive_indices, 1),
            tf.ones_like(true_positive_indices),
            shape=tf.shape(y_pred)[0:1],
        )
        false_positive_mask = tf.logical_not(
            tf.cast(true_positive_mask, tf.bool)
        )
        # Extract the appropriate confidence scores for the FP predictions.
        confidence = y_pred[:, 4]
        false_positive_confidence = tf.boolean_mask(
            confidence, false_positive_mask
        )

        # Create the ground-truth and predictions.
        positive_gt = tf.ones_like(true_positive_confidence)
        negative_gt = tf.zeros_like(false_positive_confidence)
        self._auc.update_state(positive_gt, true_positive_confidence)
        self._auc.update_state(negative_gt, false_positive_confidence)

    def result(self) -> tf.Tensor:
        return self._auc.result()

    def reset_state(self) -> None:
        self._auc.reset_state()


class FalseNegatives(_BboxMetric):
    """
    A metric that calculates the number of false negatives.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        name: str = "false_negatives",
        **kwargs: Any
    ):
        """
        Args:
            iou_threshold: The minimum IOU necessary between a ground-truth
                box and a prediction for us to consider that prediction as
                corresponding to the ground-truth.
            conf_threshold: The minimum confidence necessary for a predicted
                box to be considered as a positive.
            name: The name of this metric.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._iou_threshold = iou_threshold
        self._conf_threshold = conf_threshold

        # Keeps track of the number of false negatives.
        self._false_negatives = self.add_weight(
            name="fn", initializer="zeros", dtype=tf.int32
        )

    def _update_state_from_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> None:
        # Match ground-truth and predicted boxes.
        matches_with_confidence = _compute_bbox_matches(
            y_true, y_pred, iou_threshold=self._iou_threshold
        )

        # Find the best match for each ground-truth box.
        best_matches = tf.reduce_max(matches_with_confidence, axis=1)
        # Confidence cannot be less than 0, even if there are no matches.
        best_matches = tf.maximum(best_matches, 0.0)
        # If any true box only has matches below the confidence threshold,
        # we consider that to be a false negative.
        false_negative_mask = tf.less(best_matches, self._conf_threshold)
        num_false_negatives = tf.reduce_sum(
            tf.cast(false_negative_mask, tf.int32)
        )

        self._false_negatives.assign_add(num_false_negatives)

    def result(self) -> tf.Tensor:
        return self._false_negatives


class FalsePositives(_BboxMetric):
    """
    A metric that calculates the number of false positives.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        name: str = "false_positives",
        **kwargs: Any
    ):
        """
        Args:
            iou_threshold: The minimum IOU necessary between a ground-truth
                box and a prediction for us to consider that prediction as
                corresponding to the ground-truth.
            conf_threshold: The minimum confidence necessary for a predicted
                box to be considered as a positive.
            name: The name of this metric.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._iou_threshold = iou_threshold
        self._conf_threshold = conf_threshold

        # Keeps track of the number of false negatives.
        self._false_positives = self.add_weight(
            name="fp", initializer="zeros", dtype=tf.int32
        )

    def _update_state_from_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> None:
        # Match ground-truth and predicted boxes.
        matches_with_confidence = _compute_bbox_matches(
            y_true, y_pred, iou_threshold=self._iou_threshold
        )

        # Get the total number of ground-truth boxes with at least one match.
        # This is the number of true positives.
        is_positive = tf.greater_equal(
            matches_with_confidence, self._conf_threshold
        )
        is_true_positive = tf.greater(
            tf.reduce_sum(tf.cast(is_positive, tf.int32), axis=1), 0
        )
        num_true_positive = tf.reduce_sum(tf.cast(is_true_positive, tf.int32))

        # Subtract this from the total number of positive predictions to find
        # the number of false positives.
        confidence = y_pred[:, 4]
        num_positive_pred = tf.reduce_sum(
            tf.cast(confidence >= self._conf_threshold, tf.int32)
        )
        num_false_positives = num_positive_pred - num_true_positive

        self._false_positives.assign_add(num_false_positives)

    def result(self) -> tf.Tensor:
        return self._false_positives


class MotAccuracy(tf.keras.metrics.Metric):
    """
    Calculates the MOT Accuracy (MOTA) metric, as defined by ClearMOT.
    """

    def __init__(
        self,
        *,
        false_negatives: FalseNegatives,
        false_positives: FalsePositives,
        id_switches: IdSwitches,
        conf_threshold: float = 0.5,
        name: str = "mot_accuracy",
        **kwargs: Any
    ):
        """
        This metric reads from other existing metrics to calculate its
        results. Those metrics must be applied separately to the model. This
        metric should then by applied to the sparse bounding box output so
        it can see the total number of objects.

        Args:
            false_negatives: The `FalseNegatives` metric to read from
                internally.
            false_positives: The `FalsePositives` metric to read from
                internally.
            id_switches: The `IdSwitches` metric to read from internally.
            conf_threshold: The minimum confidence necessary for a predicted
                box to be considered as a positive.
            name: The name of this metric.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._false_negatives = false_negatives
        self._false_positives = false_positives
        self._id_switches = id_switches
        self._conf_threshold = conf_threshold

        # Keeps track of the total number of predicted objects.
        self._num_predictions = self.add_weight(
            "num_predictions", initializer="zeros", dtype=tf.int32
        )

    def update_state(
        self, _: tf.RaggedTensor, y_pred: tf.RaggedTensor, **__
    ) -> None:
        # Keep track of the total number of objects. This can be calculated
        # as the total number of positive predictions + any false negatives.
        y_pred = cast(tf.RaggedTensor, y_pred)
        confidence = y_pred[:, :, -1]
        num_predictions = tf.reduce_sum(
            tf.cast(confidence >= self._conf_threshold, tf.int32)
        )
        self._num_predictions.assign_add(num_predictions)

    def result(self) -> tf.Tensor:
        return tf.cond(
            self._num_predictions < 1,
            # If we have no objects, I guess it's technically 100% accurate...
            lambda: 1.0,
            lambda: 1.0
            - tf.cast(
                self._false_negatives.result()
                + self._false_positives.result()
                + self._id_switches.result(),
                tf.float32,
            )
            / tf.cast(
                self._num_predictions + self._false_negatives.result(),
                tf.float32,
            ),
        )


class MaxConfidence(tf.keras.metrics.Metric):
    """
    A metric that logs the maximum value in the heatmap. This can be useful
    for debugging optimization problems.
    """

    def __init__(self, name: str = "max_confidence", **kwargs: Any):
        """
        Args:
            name: The name of the metric.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(name=name, **kwargs)

        self._max_confidence = self.add_weight(
            name="max_confidence", initializer="zeros", dtype=tf.float32
        )

    def update_state(self, _, y_pred: tf.Tensor, **__) -> None:
        # Compute the maximum heatmap value.
        batch_max = tf.reduce_max(y_pred)
        self._max_confidence.assign(batch_max)

    def result(self) -> tf.Tensor:
        return self._max_confidence


def make_metrics() -> Dict[str, Any]:
    """
    Creates the metrics to use for the model.

    Returns:
        The metrics that it created.

    """
    false_negatives = FalseNegatives()
    false_positives = FalsePositives()
    id_switches = IdSwitches()
    mot_accuracy = MotAccuracy(
        false_negatives=false_negatives,
        false_positives=false_positives,
        id_switches=id_switches,
    )

    return {
        ModelTargets.GEOMETRY_SPARSE_PRED.value: [
            AveragePrecision(),
            false_negatives,
            false_positives,
            mot_accuracy,
        ],
        ModelTargets.HEATMAP.value: MaxConfidence(),
        ModelTargets.ASSIGNMENT.value: id_switches,
    }
