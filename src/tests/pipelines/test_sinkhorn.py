"""
Tests for the `sinkhorn` module.
"""


from typing import Iterable

import numpy as np
import pytest
import tensorflow as tf
import yaml
from pytest_snapshot.plugin import Snapshot

from src.cotton_flower_mot.pipelines import assignment


def test_solve_optimal_transport_obvious() -> None:
    """
    Tests that `solve_optimal_transport` works when there is an obvious
    solution.

    """
    # Arrange.
    # Set up the problem.
    cost = 1.0 - tf.eye(3)
    cost = tf.expand_dims(cost, axis=0)

    # Act.
    transport, dist = assignment.solve_optimal_transport(cost, lamb=100)
    transport = transport.numpy()
    dist = dist.numpy()

    # Assert.
    # It should have solved the problem in the obvious way.
    np.testing.assert_array_almost_equal_nulp(np.eye(3), transport)
    # The sinkhorn distance should be very small.
    assert dist == pytest.approx(0.0)


def test_solve_optimal_transport_entropy() -> None:
    """
    Tests that using different entropy values for
    `solve_optimal_transport` works as we would expect.

    """
    # Arrange.
    # Set up the problem.
    cost = 1.0 - tf.eye(3)
    cost = tf.expand_dims(cost, axis=0)

    # Act.
    transport_good, dist_good = assignment.solve_optimal_transport(
        cost, lamb=100
    )
    transport_good = transport_good.numpy()
    dist_good = dist_good.numpy()

    transport_homogeneous, dist_homogeneous = assignment.solve_optimal_transport(
        cost, lamb=0.1
    )
    transport_homogeneous = transport_homogeneous.numpy()
    dist_homogeneous = dist_homogeneous.numpy()

    # Assert.
    # Lower lambda should lead to a more homogenous solution.
    assert np.std(transport_good) > np.std(transport_homogeneous)
    # It should also have led to a worse solution.
    assert dist_good < dist_homogeneous


def test_solve_optimal_transport_deserts(snapshot: Snapshot) -> None:
    """
    Tests that `solve_optimal_transport` can solve the example "desert problem"
    from here:
    https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/

    Args:
        snapshot: The fixture to use for snapshot testing.

    """
    # Arrange.
    # Desert preferences.
    preferences = tf.constant(
        [
            [
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, -2.0, -2.0, -2.0, 2.0],
                [1.0, 2.0, 2.0, 2.0, -1.0],
                [2.0, 1.0, 0.0, 1.0, -1.0],
                [0.5, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, -1.0],
                [-2.0, 2.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 2.0, 1.0, -1.0],
            ]
        ]
    )
    # Allowed portions. (Row sums)
    portions = tf.constant([[3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 1.0]])
    # Amount of each desert. (Column sums)
    desert_amounts = tf.constant([[4.0, 2.0, 6.0, 4.0, 4.0]])

    # Act.
    transport, dist = assignment.solve_optimal_transport(
        -preferences,
        lamb=10,
        row_sums=portions,
        column_sums=desert_amounts,
    )

    # Assert.
    # Convert to a human-readable form for snapshotting.
    results = {
        "transport": transport.numpy().tolist(),
        "dist": dist.numpy().tolist(),
    }
    snapshot.assert_match(yaml.dump(results), "desert_dist.yml")


@pytest.mark.parametrize(
    ("detection_ids", "tracklet_ids", "expected_matrix"),
    [
        (
            [3, 1, 2],
            [3, 0, 4, 1],
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
        ),
        ([], [], np.empty((0, 0))),
        (
            [24, 25, 26],
            [26, 25, 24],
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ),
    ],
    ids=["nominal", "empty", "no_births_or_deaths"],
)
def test_construct_gt_sinkhorn_matrix(
    detection_ids: Iterable[int],
    tracklet_ids: Iterable[int],
    expected_matrix: np.ndarray,
) -> None:
    """
    Tests that `construct_gt_sinkhorn_matrix` works.

    Args:
        detection_ids: The list of detection IDs.
        tracklet_ids: The list of tracklet IDs.
        expected_matrix: The expected sinkhorn matrix.

    """
    # Arrange.
    detection_ids = tf.constant(detection_ids)
    tracklet_ids = tf.constant(tracklet_ids)

    # Act.
    got_matrix = assignment.construct_gt_sinkhorn_matrix(
        detection_ids=detection_ids, tracklet_ids=tracklet_ids
    ).numpy()

    # Assert.
    np.testing.assert_array_equal(
        got_matrix, expected_matrix.astype(np.float32)
    )
