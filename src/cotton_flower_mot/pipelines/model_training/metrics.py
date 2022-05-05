"""
Custom metrics for the model.
"""


from typing import Any, Dict, Optional

import keras.metrics
import tensorflow as tf

from ..dataset_io import RotNetFeatureName
from ..schemas import ModelTargets
from .graph_utils import compute_pairwise_similarities
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch
from .ragged_utils import ragged_map_fn
from .similarity_utils import compute_ious


def id_switches(
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
    switched = tf.math.logical_xor(true_assignment, pred_assignment)
    switched = tf.cast(switched, tf.int32)
    # Division by 2 is because every time the assignment changes, it results
    # in one incorrect True value and one incorrect False value.
    return tf.reduce_sum(switched, axis=1) // 2


class IdSwitches(tf.keras.metrics.Metric):
    """
    A metric that tracks the number of ID switches.
    """

    def __init__(self, name="id_switches", **kwargs: Any):
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

        _id_switches = id_switches(y_true, y_pred)

        if sample_weight is not None:
            # Weight each sample.
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, _id_switches.shape)
            _id_switches *= sample_weight

        self._num_id_switches.assign_add(tf.reduce_sum(_id_switches))

    def result(self) -> tf.Tensor:
        return self._num_id_switches


class AveragePrecision(tf.keras.metrics.Metric):
    """
    Calculates the average precision for an object detector.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        use_top_predictions: Optional[int] = 100,
        name="average_precision",
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

    def _update_auc_from_image(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Updates the internal AUC metric for a single image and its detections.

        Args:
            y_true: The true bounding boxes for the detections, with the shape
                `[num_detections, 6]`, where each row has the same layout as
                the input to `update_state`.
            y_pred: The predicted bounding boxes for the detections, with the
                shape `[num_detections, 5]`, where each row has the same
                layout as the input to `update_state`.

        Returns:
            A dummy return value, which is just a 0 tensor.

        """
        y_true = tf.ensure_shape(y_true, (None, 6))
        y_pred = tf.ensure_shape(y_pred, (None, 5))
        if self._use_top_predictions is not None:
            # Filter to top predictions.
            y_pred = self._take_top_predictions(
                y_pred, self._use_top_predictions
            )
        confidence = y_pred[:, 4]
        y_pred = y_pred[:, :4]
        # We don't need the offsets.
        y_true = y_true[:, :4]

        # Determine which predictions match up with the truth through IOU.
        ious = compute_pairwise_similarities(
            compute_ious,
            # It expects there to be a batch dimension.
            left_features=tf.expand_dims(y_true, 0),
            right_features=tf.expand_dims(y_pred, 0),
        )[0]
        # Anything below the threshold doesn't match.
        iou_matches = tf.greater_equal(ious, self._iou_threshold)

        matches_with_confidence = tf.cast(iou_matches, tf.float32) * confidence
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
        false_positive_confidence = tf.boolean_mask(
            confidence, false_positive_mask
        )

        # Create the ground-truth and predictions.
        positive_gt = tf.ones_like(true_positive_confidence)
        negative_gt = tf.zeros_like(false_positive_confidence)
        tf.print("tp_conf:", true_positive_confidence)
        tf.print("fp_conf:", false_positive_confidence)
        self._auc.update_state(positive_gt, true_positive_confidence)
        self._auc.update_state(negative_gt, false_positive_confidence)

        # Dummy return value so that map_fn works.
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
            lambda e: self._update_auc_from_image(e[0], e[1]),
            (y_true, y_pred),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int32),
        )

    def result(self) -> tf.Tensor:
        return self._auc.result()

    def reset_state(self) -> None:
        self._auc.reset_state()


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

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, **_) -> None:
        # Compute the maximum heatmap value.
        batch_max = tf.reduce_max(y_pred)
        self._max_confidence.assign(batch_max)

    def result(self) -> tf.Tensor:
        return self._max_confidence


def make_metrics() -> Dict[str, tf.keras.metrics.Metric]:
    """
    Creates the metrics to use for the model.

    Returns:
        The metrics that it created.

    """
    # return {ModelTargets.ASSIGNMENT.value: IdSwitches()}
    return {
        ModelTargets.GEOMETRY_SPARSE_PRED.value: AveragePrecision(),
        ModelTargets.HEATMAP.value: MaxConfidence(),
    }


def make_rotnet_metrics() -> Dict[str, tf.keras.metrics.Metric]:
    """
    Creates metrics to use for RotNet pre-training.

    Returns:
        The metrics that it created.

    """
    return {
        RotNetFeatureName.PREDICTIONS.value: keras.metrics.SparseCategoricalAccuracy()
    }
