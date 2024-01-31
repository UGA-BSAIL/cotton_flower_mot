"""
Tests for the `graph_utils` module.
"""


import itertools
from typing import Iterable, Tuple

import numpy as np
import pytest
import spektral
from faker import Faker

from src.cotton_flower_mot import graph_utils


def _iter_feature_pairs(
    edge_features: np.ndarray,
) -> Iterable[Tuple[np.ndarray]]:
    """
    Iterates through pairs of node features in the edge feature tensor.

    Args:
        edge_features: The edge features.

    Yields:
        Each pair of node features.

    """
    _batch_size, num_left, num_right, _, _ = edge_features.shape

    for b, l, r in itertools.product(
        *map(range, [_batch_size, num_left, num_right])
    ):
        feature_pair = edge_features[b][l][r]
        yield feature_pair[0], feature_pair[1]


@pytest.mark.parametrize(
    ("batch_size", "num_left_nodes", "num_right_nodes", "num_features"),
    [(1, 2, 3, 4), (3, 2, 3, 4), (3, 5, 5, 5), (3, 4, 4, 1), (3, 0, 4, 2)],
    ids=(
        "batch_size_1",
        "different_num_nodes",
        "same_num_nodes",
        "feature_size_1",
        "no_left_nodes",
    ),
)
def test_compute_bipartite_edge_features(
    faker: Faker,
    batch_size: int,
    num_left_nodes: int,
    num_right_nodes: int,
    num_features: int,
) -> None:
    """
    Tests that `compute_bipartite_edge_features` works.

    Args:
        faker: The fixture to use for generating fake data.
        batch_size: The batch size to use for testing.
        num_left_nodes: The number of nodes on the left side of the graph.
        num_right_nodes: The number of nodes on the right side of the graph.
        num_features: The length of the node feature vectors.

    """
    # Arrange.
    # Create two sets of features.
    left_nodes = faker.tensor((batch_size, num_left_nodes, num_features))
    right_nodes = faker.tensor((batch_size, num_right_nodes, num_features))

    # Act.
    edge_features = graph_utils.compute_bipartite_edge_features(
        left_nodes, right_nodes
    ).numpy()

    # Assert.
    # Make sure the shape is correct.
    assert edge_features.shape == (
        batch_size,
        num_left_nodes,
        num_right_nodes,
        2,
        num_features,
    )

    # Make sure we have all possible pairs of features.
    left_nodes = left_nodes.numpy()
    right_nodes = right_nodes.numpy()

    for (b, l, r), (expected_left, expected_right) in zip(
        itertools.product(
            *map(range, [batch_size, num_left_nodes, num_right_nodes]),
        ),
        _iter_feature_pairs(edge_features),
    ):
        left_feature = left_nodes[b][l]
        right_feature = right_nodes[b][r]
        np.testing.assert_allclose(left_feature, expected_left)
        np.testing.assert_allclose(right_feature, expected_right)


@pytest.mark.parametrize(
    ("batch_size", "num_left", "num_right", "num_features"),
    [(3, 4, 5, 3), (3, 4, 5, 1), (1, 4, 5, 3), (3, 4, 4, 3)],
    ids=(
        "different_num_nodes",
        "feature_size_1",
        "batch_size_1",
        "same_num_nodes",
    ),
)
def test_make_adjacency_matrix(
    faker: Faker,
    batch_size: int,
    num_left: int,
    num_right: int,
    num_features: int,
) -> None:
    """
    Tests that `make_adjacency_matrix` works.

    Args:
        faker: The fixture to use for generating fake data.
        batch_size: The batch size to use for testing.
        num_left: The number of left nodes to use for testing.
        num_right: The number of right nodes to use for testing.
        num_features: The feature length to use for testing.

    """
    # Arrange.
    # Create features to test with.
    edge_features = faker.tensor(
        (batch_size, num_left, num_right, num_features)
    )

    # Act.
    affinity_matrix = graph_utils.make_adjacency_matrix(edge_features).numpy()

    # Assert.
    # It should have the correct shape.
    num_nodes = num_left + num_right
    assert affinity_matrix.shape == (
        batch_size,
        num_nodes,
        num_nodes,
        num_features,
    )

    # It should match the input features for the bipartite edges.
    for b, l, r in itertools.product(
        *map(range, [batch_size, num_left, num_right])
    ):
        np.testing.assert_array_almost_equal(
            edge_features[b][l][r], affinity_matrix[b][l][num_left + r]
        )

    # It should have zeros for non bipartite edges.
    zero_feature = np.zeros((num_features,))
    for b, l, r in itertools.product(
        *map(range, [batch_size, num_left, num_right])
    ):
        np.testing.assert_array_almost_equal(
            zero_feature, affinity_matrix[b][num_left + l][r]
        )


@pytest.mark.parametrize(
    ["batch_size", "num_nodes", "num_edge_features", "num_node_features"],
    [(1, 16, 10, 20), (5, 16, 1, 30), (8, 16, 10, 20)],
    ids=["single_batch", "single_edge_feature", "multiple_edge_features"],
)
def test_augment_adjacency_matrix(
    faker: Faker,
    batch_size: int,
    num_nodes: int,
    num_edge_features: int,
    num_node_features: int,
) -> None:
    """
    Tests that `augment_adjacency_matrix` works.

    Args:
        faker: The fixture to use for generating fake data.
        batch_size: The batch size to use for the fake data.
        num_nodes: The number of nodes to use for the fake data.
        num_edge_features: The number of edge features to use.
        num_node_features: The number of node features to use.

    """
    # Arrange.
    # Create fake edge and node features.
    node_features = faker.tensor((batch_size, num_nodes, num_node_features))
    edge_features = faker.tensor(
        (batch_size, num_nodes, num_nodes, num_edge_features)
    )

    # Act.
    # Create the augmented adjacency matrix.
    augmented_edge_features = graph_utils.augment_adjacency_matrix(
        adjacency_matrix=edge_features, node_features=node_features
    ).numpy()

    # Assert.
    # Make sure we get the right concatenated feature at each point.
    for batch, row, col in itertools.product(
        *map(range, [batch_size, num_nodes, num_nodes])
    ):
        expected_feature = np.concatenate(
            [
                edge_features[batch][row][col].numpy(),
                node_features[batch][col].numpy(),
                node_features[batch][row].numpy(),
            ],
            axis=0,
        )
        actual_feature = augmented_edge_features[batch][row][col]

        np.testing.assert_array_max_ulp(expected_feature, actual_feature)


@pytest.mark.parametrize("power", [-0.5, -1.0, 0.5])
def test_degree_power(faker: Faker, power: float) -> None:
    """
    Tests that `degree_power` works as a substitute for the Spektral version.

    Args:
        faker: The fixture to use for generating fake data.
        power: The power to try raising it to.

    """
    # Arrange.
    # Create a tensor to test with. Don't include negatives since in real life,
    # the adjacency matrix should always be positive.
    adjacency = faker.tensor((10, 8, 8), min_value=0.0)
    adjacency_np = adjacency.numpy()

    # Act.
    # Compute with our version.
    tensor_result = graph_utils.degree_power(adjacency, power).numpy()
    # Compute with the Spektral version.
    spektral_result = np.empty_like(adjacency_np)
    for i in range(adjacency_np.shape[0]):
        spektral_result[i] = spektral.utils.degree_power(
            adjacency_np[i], power
        )

    # Assert.
    # They should be equivalent.
    np.testing.assert_array_max_ulp(spektral_result, tensor_result, 2)


@pytest.mark.parametrize(
    "symmetric", [True, False], ids=["symmetric", "not_symmetric"]
)
def test_normalized_adjacency(faker: Faker, symmetric: bool) -> None:
    """
    Tests that `normalized_adjacency` works as a substitute for the Spektral
    version.

    Args:
        faker: The fixture to use for generating fake data.
        symmetric: Whether to test symmetric normalization.

    """
    # Arrange.
    # Create a tensor to test with. Don't include negatives since in real life,
    # the adjacency matrix should always be positive.
    adjacency = faker.tensor((10, 8, 8), min_value=0.0)
    adjacency_np = adjacency.numpy()

    # Act.
    # Compute with our version.
    tensor_result = graph_utils.normalized_adjacency(
        adjacency, symmetric=symmetric
    ).numpy()
    # Compute with the Spektral version.
    spektral_result = np.empty_like(adjacency_np)
    for i in range(adjacency_np.shape[0]):
        spektral_result[i] = spektral.utils.normalized_adjacency(
            adjacency_np[i], symmetric=symmetric
        )

    # Assert.
    # They should be equivalent.
    np.testing.assert_array_max_ulp(spektral_result, tensor_result, 3)


@pytest.mark.parametrize(
    "symmetric", [True, False], ids=["symmetric", "not_symmetric"]
)
def test_gcn_filter(faker: Faker, symmetric: bool) -> None:
    """
    Tests that `gcn_filter` works as a substitute for the Spektral version.

    Args:
        faker: The fixture to use for generating fake data.
        symmetric: Whether to test symmetric normalization.

    """
    # Arrange.
    # Create a tensor to test with. Don't include negatives since in real life,
    # the adjacency matrix should always be positive.
    adjacency = faker.tensor((10, 8, 8), min_value=0.0)
    adjacency_np = adjacency.numpy()

    # Act.
    # Compute with our version.
    tensor_result = graph_utils.gcn_filter(
        adjacency, symmetric=symmetric
    ).numpy()
    # Compute with the Spektral version.
    spektral_result = spektral.utils.gcn_filter(
        adjacency_np, symmetric=symmetric
    )

    # Assert.
    # They should be equivalent.
    np.testing.assert_array_max_ulp(spektral_result, tensor_result, 3)


def test_bound_adjacency(faker: Faker) -> None:
    """
    Tests that `bound_adjacency` works.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    # Create a fake adjacency matrix, which might have negative values.
    adjacency = faker.tensor((10, 8, 8, 1))

    # Act.
    normalized = graph_utils.bound_adjacency(adjacency).numpy()

    # Assert.
    # All values should be positive.
    assert normalized.min() >= 0.0
