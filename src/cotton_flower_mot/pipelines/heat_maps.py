"""
Utilities for creating the heat maps used by CenterNet and related
keypoint-based detectors.
"""

import math
from typing import Any, Optional

import tensorflow as tf


def trim_out_of_bounds(points: tf.Tensor) -> tf.Tensor:
    """
    Convenience function to trim points that are not within `[0, 1]`. It is
    invoked automatically when creating heatmaps.

    Args:
        points: The points to trim, as a matrix of shape
            `[num_points, num_dims]`. It will assume that the first two
            dimensions are the x and y coordinates.

    Returns:
        The trimmed points.

    """
    x_valid = tf.logical_and(points[:, 0] >= 0.0, points[:, 0] <= 1.0)
    y_valid = tf.logical_and(points[:, 1] >= 0.0, points[:, 1] <= 1.0)
    # Both coordinates of a point must be valid to keep that point.
    point_valid = tf.logical_and(x_valid, y_valid)
    return tf.boolean_mask(points, point_valid)


def _to_pixel_points(points: tf.Tensor, *, map_size: tf.Tensor) -> tf.Tensor:
    """
    Converts points from normalized to pixel coordinates.

    Args:
        points: The normalized points to convert.
        map_size: The size of the output heatmap.

    Returns:
        The points in pixel coordinates, with out-of-bounds points excluded.

    """
    points = tf.ensure_shape(points, (None, 2))
    map_size = tf.ensure_shape(map_size, (2,))
    points = tf.cast(points, tf.float32)

    # Eliminate any points that are out-of-bounds.
    points = trim_out_of_bounds(points)

    # Quantize the annotations to convert from frame fractions to actual
    # pixel values.
    pixel_points = points * tf.cast(map_size - tf.constant(1), tf.float32)
    return tf.cast(tf.round(pixel_points), tf.int64)


def _max_overlap_radii(
    bbox_sizes: tf.Tensor, min_iou: float = 0.3
) -> tf.Tensor:
    """
    Calculates the radii that our center predictions needs to be within of
    the truth in order to achieve some minimum IOU.

    Args:
        bbox_sizes: The sizes of the bounding boxes. Should have shape
            `[num_boxes, 2]`, where the second dimension is of the form
            `[width, height]`.
        min_iou: The minimum IOU we want.

    Returns:
        The radii corresponding to each box.

    """
    bbox_width = tf.cast(bbox_sizes[:, 0], tf.float32)
    bbox_height = tf.cast(bbox_sizes[:, 1], tf.float32)

    a1 = 1.0
    b1 = bbox_height + bbox_width
    c1 = bbox_width * bbox_height * (1.0 - min_iou) / (1.0 + min_iou)
    sq1 = tf.sqrt(b1**2.0 - 4.0 * a1 * c1)
    r1 = (b1 + sq1) / 2.0
    a2 = 4.0
    b2 = 2.0 * (bbox_height + bbox_width)
    c2 = (1.0 - min_iou) * bbox_width * bbox_height
    sq2 = tf.sqrt(b2**2.0 - 4.0 * a2 * c2)
    r2 = (b2 + sq2) / 2.0
    a3 = 4.0 * min_iou
    b3 = -2.0 * min_iou * (bbox_height + bbox_width)
    c3 = (min_iou - 1.0) * bbox_width * bbox_height
    sq3 = tf.sqrt(b3**2.0 - 4.0 * a3 * c3)
    r3 = (b3 + sq3) / 2.0
    return tf.minimum(tf.minimum(r1, r2), r3)


def _compute_hash(data: tf.Tensor) -> tf.Tensor:
    """
    Computes the hashes of arbitrary data.

    Args:
        data: The data to hash. The first dimension is the batch dimension,
            and the others will be hashed.

    Returns:
        A vector with the same length as the batch, with one hash value for
        each element.

    """
    fingerprint = tf.fingerprint(data)

    # Combine the bytes into a single number.
    fingerprint = tf.cast(fingerprint, tf.uint64)
    shifted = []
    for i, shift in enumerate(range(0, 64, 8)):
        shifted.append(tf.bitwise.left_shift(fingerprint[:, i], shift))
    return sum(shifted)


def _de_duplicate_points(points: tf.Tensor) -> tf.Tensor:
    """
    Removes duplicate values from a set of points.

    Args:
        points: The points to process. Should be in the form (x, y).

    Returns:
        The same points, but without duplicates.

    """
    # Hash all the points.
    point_hashes = _compute_hash(points)
    # Find unique values.
    _, indices = tf.unique(point_hashes)
    unique_indices, _ = tf.unique(indices)

    # Limit to only the unique points.
    return tf.gather(points, unique_indices, axis=0)


def make_point_annotation_map(
    points: tf.Tensor,
    *,
    map_size: tf.Tensor,
    point_values: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Creates dense point maps given a set of point locations. Each point
    corresponds to a pixel in the map that will be set to a non-zero value,
    while the rest will be set to zero.

    Args:
        points: The points to add to the map. Should be in the form (x, y).
        map_size: The size of the map, in the form (width, height).
        point_values: If specified, these are the corresponding values which
            will be set at the location of each point. Should be a 1D tensor
            with the same length as `points`. If not specified, all values will
            be set to one.

    Returns:
        The point annotation map that it created. It will have the shape
        specified by `map_size`.

    """
    pixel_points = _to_pixel_points(points, map_size=map_size)
    pixel_points = _de_duplicate_points(pixel_points)

    # Generate the output maps.
    if point_values is None:
        # Use ones for all the values.
        num_non_zero_values = tf.shape(pixel_points)[0]
        point_values = tf.ones((num_non_zero_values,), dtype=tf.float32)
    sparse_maps = tf.SparseTensor(
        indices=pixel_points[:, ::-1],
        values=point_values,
        # See https://github.com/tensorflow/tensorrt/issues/118 for
        # explanation of indexing.
        dense_shape=tf.cast(map_size[..., ::-1], dtype=tf.int64),
    )
    # Reorder indices to conform with sparse tensor conventions.
    sparse_maps = tf.sparse.reorder(sparse_maps)

    dense = tf.sparse.to_dense(
        sparse_maps,
        default_value=0.0,
    )
    return dense


def make_heat_map(
    points: tf.Tensor,
    *,
    sigmas: tf.Tensor,
    map_size: tf.Tensor,
    normalized: bool = True,
) -> tf.Tensor:
    """
    Creates a gaussian heat map by splatting a set of points.

    Args:
        points: The points to add to the map. Should be in the form (x, y).
            They should be normalized to [0.0, 1.0]
        map_size: The size of the map in pixels, in the form (width, height).
        sigmas: The corresponding standard deviation to use for each point.
            Should be a 1D vector.
        normalized: If true, the gaussians will be normalized, such that
            the integral for each adds up to one. Otherwise, they will be
            un-normalized.

    Returns:
        A tensor containing heat map, of the shape
        (height, width, 1)

    """
    points = tf.ensure_shape(points, (None, 2))
    sigmas = tf.cast(tf.ensure_shape(sigmas, (None,)), tf.float32)
    map_size = tf.ensure_shape(map_size, (2,))

    pixel_points = _to_pixel_points(points, map_size=map_size)
    map_size = tf.cast(map_size, tf.int64)

    # Create grids to apply the gaussians on.
    grid_indices_x, grid_indices_y = tf.meshgrid(
        tf.range(map_size[0]),
        tf.range(map_size[1]),
    )
    # We want a separate grid for each point.
    num_points = tf.shape(points, out_type=tf.int64)[0]
    heatmap_shape = tf.concat(
        (map_size[..., ::-1], tf.expand_dims(num_points, 0)), axis=0
    )
    grid_indices_x = tf.broadcast_to(
        tf.expand_dims(grid_indices_x, 2), heatmap_shape
    )
    grid_indices_y = tf.broadcast_to(
        tf.expand_dims(grid_indices_y, 2), heatmap_shape
    )

    # Calculate the gaussian.
    point_x = tf.cast(pixel_points[:, 0], tf.float32)
    point_y = tf.cast(pixel_points[:, 1], tf.float32)
    grid_indices_x = tf.cast(grid_indices_x, tf.float32)
    grid_indices_y = tf.cast(grid_indices_y, tf.float32)
    gaussian_values = tf.exp(
        -(
            tf.square(grid_indices_x - point_x)
            + tf.square(grid_indices_y - point_y)
        )
        / (2.0 * tf.square(sigmas))
    )
    if normalized:
        gaussian_values *= 1.0 / (2.0 * math.pi * tf.square(sigmas))

    # Convert back to heatmaps.
    heatmaps = tf.reshape(gaussian_values, heatmap_shape)
    # Squash into a single heatmap.
    return tf.reduce_max(heatmaps, axis=2, keepdims=True)


def make_object_heat_map(
    boxes: tf.Tensor,
    *,
    map_size: tf.Tensor,
    min_iou: float = 0.3,
    **kwargs: Any,
) -> tf.Tensor:
    """
    Generates a heatmap with a dynamic sigma value chosen for each point
    based on the size of the object.

    Args:
        boxes: The object bounding boxes. Should be a 2D matrix where each row
            has the form `[center_x, center_y, width, height]`, and all
            coordinates are normalized.
        map_size: The size of the map in pixels, in the form (width, height).
        min_iou: The minimum IOU we want for a correctly-sized box prediction
            that is within three sigma of the ground truth.
        **kwargs: Additional arguments will be forwarded to `make_heat_map()`.

    Returns:
        A tensor containing heat map, of the shape
        (height, width, 1)

    """
    boxes = tf.ensure_shape(boxes, (None, 4))
    map_size = tf.ensure_shape(map_size, (2,))

    # Calculate the sigmas to use.
    pixel_sizes = boxes[:, 2:] * tf.cast(map_size, tf.float32)
    radii = _max_overlap_radii(pixel_sizes, min_iou=min_iou)
    sigmas = radii / tf.constant(3.0)

    # Create the heat maps.
    centers = boxes[:, :2]
    return make_heat_map(centers, sigmas=sigmas, map_size=map_size, **kwargs)
