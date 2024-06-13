import enum
from typing import Tuple

import networkx as nx
import numpy as np
import pytest
from keras import Input, Model
import tensorflow as tf

from src.cotton_flower_mot.pipelines.model_training.layers.cens_net import (
    CensNet,
)

batch_size = 32
N = 11
F = 7
S = 3
A = np.ones((N, N))

NODE_CHANNELS = 8
"""
Number of node output channels to use for testing.
"""
EDGE_CHANNELS = 10
"""
Number of edge output channels to use for testing.
"""


@enum.unique
class Mode(enum.IntEnum):
    """
    Represents the data modes to use for testing.
    """

    SINGLE = enum.auto()
    BATCH = enum.auto()
    MIXED = enum.auto()


GraphDescriptors = Tuple[np.array, np.array, np.array]
"""
Describes a graph to use for testing.
"""


@pytest.fixture()
def random_graph_descriptors() -> GraphDescriptors:
    """
    Creates a random graph to use, and computes its various descriptors.
    :return: The adjacency matrices for the graph and line graph, and the
        incidence matrix.
    """
    graph = nx.dense_gnm_random_graph(11, 50, seed=1337)
    line_graph = nx.line_graph(graph)
    node_adjacency = nx.to_numpy_array(graph)
    edge_adjacency = nx.to_numpy_array(line_graph)
    incidence = np.array(nx.incidence_matrix(graph).todense())

    node_adjacency += np.eye(node_adjacency.shape[0])
    edge_adjacency += np.eye(edge_adjacency.shape[0])

    return node_adjacency, edge_adjacency, incidence


def test_smoke(random_graph_descriptors: GraphDescriptors) -> None:
    """
    Tests that we can create a model with the layer, and it processes
    input data without crashing.

    Args:
        random_graph_descriptors: Descriptors for the graph to use when
            testing.
    """
    # Arrange.
    node_adjacency, edge_adjacency, incidence = random_graph_descriptors
    node_adjacency = tf.sparse.from_dense(node_adjacency)
    edge_adjacency = tf.sparse.from_dense(edge_adjacency)
    incidence = tf.sparse.from_dense(incidence)

    # Create node and edge features.
    node_feature_shape = (node_adjacency.shape[0], F)
    edge_feature_shape = (edge_adjacency.shape[0], S)

    node_features = tf.random.normal(shape=node_feature_shape)
    edge_features = tf.random.normal(shape=edge_feature_shape)

    # Create the model.
    node_input = Input(shape=node_features.shape[1:])
    node_adjacency_input = Input(shape=node_adjacency.shape[1:], sparse=True)
    edge_adjacency_input = Input(shape=edge_adjacency.shape[1:], sparse=True)
    incidence_input = Input(shape=incidence.shape[1:], sparse=True)
    edge_input = Input(shape=edge_features.shape[1:])

    next_nodes, next_edges = CensNet(
        num_nodes_out=NODE_CHANNELS,
        num_edges_out=EDGE_CHANNELS,
        activation="relu",
    )(
        (
            node_input,
            (node_adjacency_input, edge_adjacency_input, incidence_input),
            edge_input,
        )
    )

    model = Model(
        inputs=(
            node_input,
            edge_input,
            node_adjacency_input,
            edge_adjacency_input,
            incidence_input,
        ),
        outputs=(next_nodes, next_edges),
    )

    # Act.
    # Run the model.
    got_next_nodes, got_next_edges = model(
        [
            node_features,
            edge_features,
            node_adjacency,
            edge_adjacency,
            incidence,
        ]
    )

    # Assert.
    # Make sure that the output shapes are correct.
    got_node_shape = got_next_nodes.numpy().shape
    got_edge_shape = got_next_edges.numpy().shape
    assert got_node_shape == node_features.shape[:-1] + (NODE_CHANNELS,)
    assert got_edge_shape == edge_features.shape[:-1] + (EDGE_CHANNELS,)


def test_get_config_round_trip():
    """
    Tests that it is possible to serialize a layer using `get_config()`,
    and then re-instantiate an identical one.
    """
    # Arrange.
    # Create the layer to test with.
    layer = CensNet(num_nodes_out=NODE_CHANNELS, num_edges_out=EDGE_CHANNELS)

    # Act.
    config = layer.get_config()
    new_layer = CensNet(**config)

    # Assert.
    # The new layer should be the same.
    assert new_layer.num_nodes_out == layer.num_nodes_out
    assert new_layer.num_edges_out == layer.num_edges_out


def test_preprocess_smoke():
    """
    Tests that the preprocessing functionality does not crash.
    """
    # Act.
    node_adjacency, edge_adjacency, incidence = CensNet.preprocess(
        tf.sparse.from_dense(A)
    )

    # Assert.
    node_adjacency = tf.sparse.to_dense(node_adjacency).numpy()
    edge_adjacency = tf.sparse.to_dense(edge_adjacency).numpy()
    incidence = tf.sparse.to_dense(incidence).numpy()

    # All matrices should be binary.
    assert np.all(node_adjacency >= 0)
    assert np.all(node_adjacency <= 1)
    assert np.all(edge_adjacency >= 0)
    assert np.all(edge_adjacency <= 1)
    assert np.all(incidence >= 0)
    assert np.all(incidence <= 1)

    # It should have self-loops.
    assert np.all(np.diag(node_adjacency) == 1)
    assert np.all(np.diag(edge_adjacency) == 1)
