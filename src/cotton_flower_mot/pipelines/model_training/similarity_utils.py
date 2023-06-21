"""
Utilities for computing similarity metrics.
"""


import math
from typing import Callable, Tuple
from ..training_utils import bound_numerics

import tensorflow as tf

_EPSILON = tf.constant(1.0e-6)
"""
Small value to use to avoid division by zero.
"""


def _min_max(boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes minimum and maximum extents for bounding boxes.

    Args:
        boxes: The bounding boxes.

    Returns:
        The minimum points and maximum points of the boxes.

    """
    boxes_center = boxes[:, :2]
    boxes_half_size = boxes[:, 2:] / 2.0

    boxes_min = boxes_center - boxes_half_size
    boxes_max = boxes_center + boxes_half_size
    return boxes_min, boxes_max


def _sorted_box_boundaries(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Computes the boundary points for two sets of boxes, sorted along each axis.
    This is an intermediate step in IOU computations.

    Args:
        boxes1: First array of bounding boxes. It should have the shape
            `[n_boxes, 4]`, where the columns are the horizontal position
            of the box center, the vertical position of the box center,
            the box's width, and the box's height, in that order.
        boxes2: The second array of bounding boxes. Has the same shape and
            format as `boxes_pred`.

    Returns:
        An array of shape `[num_boxes, 4, 2]`, where the second dimension is
        the sorted one and the last dimension is the two axes.

    """
    boxes1 = tf.ensure_shape(boxes1, (None, 4))
    boxes2 = tf.ensure_shape(boxes2, (None, 4))

    boxes1_rank2 = tf.assert_rank(boxes1, 2)
    boxes2_rank2 = tf.assert_rank(boxes2, 2)

    with tf.control_dependencies([boxes1_rank2, boxes2_rank2]):
        boxes1_min, boxes1_max = _min_max(boxes1)
        boxes2_min, boxes2_max = _min_max(boxes2)

    box_boundaries = tf.stack(
        [boxes1_min, boxes1_max, boxes2_min, boxes2_max], axis=1
    )
    return tf.sort(box_boundaries, axis=1)


def compute_ious(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Computes the IOUs between two sets of bounding boxes.

    Args:
        boxes1: First array of bounding boxes. It should have the shape
            `[n_boxes, 4]`, where the columns are the horizontal position
            of the box center, the vertical position of the box center,
            the box's width, and the box's height, in that order.
        boxes2: The second array of bounding boxes. Has the same shape and
            format as `boxes_pred`.

    Returns:
        An array of shape `[n_boxes,]` containing the IOUs between each pair
        of bounding boxes.

    """
    sorted_boundaries = _sorted_box_boundaries(boxes1, boxes2)

    # Extract the intersection and union.
    intersection = sorted_boundaries[:, 2, :] - sorted_boundaries[:, 1, :]
    # Size of larger box that encloses both boxes.
    enclosing = sorted_boundaries[:, 3, :] - sorted_boundaries[:, 0, :]

    # Account for the case where the boxes are disjoint.
    boxes1_sizes = boxes1[:, 2:]
    boxes2_sizes = boxes2[:, 2:]
    total_size = boxes1_sizes + boxes2_sizes
    is_disjoint = enclosing > total_size
    # If they are disjoint, the intersection is zero.
    zero = tf.constant(0.0, dtype=intersection.dtype)
    intersection = tf.where(is_disjoint, zero, intersection)

    # Figure out the intersection area.
    intersection_area = tf.reduce_prod(intersection, axis=1)
    # The union area is simply the sum of the areas of both boxes - the
    # intersection area.
    box1_area = tf.reduce_prod(boxes1_sizes, axis=1)
    box2_area = tf.reduce_prod(boxes2_sizes, axis=1)
    union_area = box1_area + box2_area - intersection_area

    epsilon = tf.cast(_EPSILON, union_area.dtype)
    return intersection_area / tf.maximum(union_area, epsilon)


def distance_penalty(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Computes the normalized distance between two sets of bounding boxes, as
    specified by https://arxiv.org/pdf/1911.08287.pdf. This is the penalty
    term in the dIOU loss calculation.

    Args:
        boxes1: First array of bounding boxes. It should have the shape
            `[n_boxes, 4]`, where the columns are the horizontal position
            of the box center, the vertical position of the box center,
            the box's width, and the box's height, in that order.
        boxes2: The second array of bounding boxes. Has the same shape and
            format as `boxes_pred`.

    Returns:
        An array of shape `[n_boxes,]` containing the normalized distance
        between each pair of bounding boxes centers.

    """
    # Compute the diagonal length of the enclosing box.
    sorted_boundaries = _sorted_box_boundaries(boxes1, boxes2)
    enclosing_size = sorted_boundaries[:, 3, :] - sorted_boundaries[:, 0, :]
    diagonal_length = tf.norm(enclosing_size, axis=1)

    # Compute the distance between the box centers.
    boxes1_centers = boxes1[:, :2]
    boxes2_centers = boxes2[:, :2]
    center_distance = tf.norm(boxes2_centers - boxes1_centers, axis=1)

    epsilon = tf.cast(_EPSILON, diagonal_length.dtype)
    return tf.square(center_distance) / tf.maximum(
        tf.square(diagonal_length), epsilon
    )


def aspect_ratio_penalty(
    boxes_pred: tf.Tensor, boxes_gt: tf.Tensor
) -> tf.Tensor:
    """
    Computes the aspect ratio penalty term, as used in the cIOU loss
    calculation from https://arxiv.org/pdf/1911.08287.pdf.

    Args:
        boxes_pred: Array of prediction bounding boxes. It should have the
            shape `[n_boxes, 4]`, where the columns are the horizontal position
            of the box center, the vertical position of the box center,
            the box's width, and the box's height, in that order.
        boxes_gt: The array of GT bounding boxes. Has the same shape and
            format as `boxes1`.

    Returns:
        An array of shape `[n_boxes,]` containing the cIOU aspect ratio
        penalty for each pair of bounding boxes.

    """
    # Tensorflow doesn't support atan2 on 16-bit floats, so everything has to
    # be 32-bit here.
    original_dtype = boxes_pred.dtype
    boxes_pred = tf.cast(boxes_pred, tf.float32)
    boxes_gt = tf.cast(boxes_gt, tf.float32)

    widths_pred = boxes_pred[:, 2]
    widths_gt = boxes_gt[:, 2]
    heights_pred = boxes_pred[:, 3]
    heights_gt = boxes_gt[:, 3]

    # Ground-truth values should be constant.
    widths_gt = tf.stop_gradient(widths_gt)
    heights_gt = tf.stop_gradient(heights_gt)

    @tf.custom_gradient
    def _consistency_parameter(
        _widths_pred: tf.Tensor, _heights_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple]]:
        def _grad(dv: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            # Use an approximation of the gradient for numerical stability.
            scale = dv * tf.constant(8.0) / tf.square(math.pi)
            angle_diff = tf.atan2(widths_gt, heights_gt) - tf.atan2(
                _widths_pred, _heights_pred
            )
            d_width = scale * angle_diff * _heights_pred
            d_height = -scale * angle_diff * _widths_pred

            return d_width, d_height

        # Calculate aspect ratio consistency parameter.
        return (
            4.0
            / tf.square(math.pi)
            * tf.square(
                tf.atan2(widths_gt, heights_gt)
                - tf.atan2(_widths_pred, _heights_pred)
            )
        ), _grad

    # Calculate the trade-off parameter.
    v = _consistency_parameter(widths_pred, heights_pred)
    alpha = v / tf.maximum(
        (tf.constant(1.0) - compute_ious(boxes_pred, boxes_gt) + v),
        _EPSILON,
    )

    return tf.cast(alpha * v, original_dtype)


def cosine_similarity(features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:
    """
    Computes the cosine similarity between two sets of feature vectors.

    Args:
        features1: The first set of feature vectors, with shape
            `[batch_size, n_features]`.
        features2: The second set of feature vectors, with shape
            `[batch_size, n_features]`.

    Returns:
        The cosine similarities between corresponding features, with shape
        `[batch_size,]`.

    """
    original_dtype = features1.dtype
    features1 = tf.cast(features1, tf.float32)
    features2 = tf.cast(features2, tf.float32)
    # features1 = bound_numerics(features1)
    # features2 = bound_numerics(features2)

    feature_dot = tf.reduce_sum(features1 * features2, axis=-1)
    feature1_mag = tf.norm(features1, axis=-1)
    feature2_mag = tf.norm(features2, axis=-1)
    denominator = feature1_mag * feature2_mag

    feature_dot = tf.debugging.assert_all_finite(feature_dot, "feature_dot")
    denominator = tf.debugging.assert_all_finite(denominator, "denominator")
    similarity = feature_dot / tf.maximum(denominator, _EPSILON)

    # Convert back to original dtype.
    similarity = tf.clip_by_value(
        similarity, original_dtype.min, original_dtype.max
    )
    return tf.cast(similarity, original_dtype)
