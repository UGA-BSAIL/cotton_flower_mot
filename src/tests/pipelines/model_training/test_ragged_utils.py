"""
Tests for the `ragged_utils` module.
"""

import numpy as np
import tensorflow as tf

from src.cotton_flower_mot.pipelines.model_training import ragged_utils


def test_ragged_map_fn() -> None:
    """
    Tests that `ragged_map_fn` works in a basic fashion.

    """
    # Arrange.
    ragged_input = tf.RaggedTensor.from_row_lengths(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 2, 3, 2, 1, 2]
    )
    expected_result = tf.constant([0, 3, 12, 13, 8, 19])

    # Act.
    sums = ragged_utils.ragged_map_fn(
        tf.reduce_sum,
        ragged_input,
        fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int32),
    )

    # Assert.
    np.testing.assert_array_equal(expected_result.numpy(), sums.numpy())


def test_ragged_map_fn_nested() -> None:
    """
    Tests that `ragged_map_fn` works when we have a nested input.

    """
    # Arrange.
    ragged_input = tf.RaggedTensor.from_row_lengths(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 2, 3, 2, 1, 2]
    )
    normal_input = tf.range(0, limit=6)
    expected_result = tf.constant([0, 4, 14, 16, 12, 24])

    # Act.
    sums = ragged_utils.ragged_map_fn(
        lambda e: tf.reduce_sum(e[0]) + e[1],
        (ragged_input, normal_input),
        fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int32),
    )

    # Assert.
    np.testing.assert_array_equal(expected_result.numpy(), sums.numpy())
