"""
Selection of utilities for dealing with graphs.
"""


from typing import Callable, Union

import tensorflow as tf

_EPSILON = tf.constant(0.0001)
"""
Small value to use to avoid dividing by zero.
"""


def compute_bipartite_edge_features(
    left_nodes: tf.Tensor, right_nodes: tf.Tensor
) -> tf.Tensor:
    """
    Computes edge features for a bipartite graph connecting nodes in
    `left_nodes` to nodes in `right_nodes`. Each node in `left_node` will
    be assumed to have an edge to every node in `right_nodes`.

    Args:
        left_nodes: Node features for the left nodes. This should be a 3D tensor
            with shape [batch, n_left_nodes, n_features].
        right_nodes: Node features for the right nodes. This should be a 3D
            tensor with shape [batch, n_right_nodes, n_features].

    Returns:
        A 5D tensor that enumerates every possible combination of a left node
        with a right node. The shape would end up being
        [batch, n_left_nodes, n_right_nodes, 2, n_features].

        The last two dimensions contain the original features from the left node
        and right node.

        The features should be enumerated in row-major order, assuming that
        left nodes are rows and right nodes are columns.

    """
    left_rank_3 = tf.assert_rank(left_nodes, 3)
    right_rank_3 = tf.assert_rank(right_nodes, 3)

    with tf.control_dependencies([left_rank_3, right_rank_3]):
        # Compute the node dimension index pairs.
        num_left_nodes = tf.shape(left_nodes)[1]
        num_right_nodes = tf.shape(right_nodes)[1]
        num_features = tf.shape(left_nodes)[2]
        left_node_indices = tf.range(0, limit=num_left_nodes)
        right_node_indices = tf.range(0, limit=num_right_nodes)

        index_pair_right, index_pair_left = tf.meshgrid(
            right_node_indices, left_node_indices
        )
        index_pair_left = tf.reshape(index_pair_left, (-1,))
        index_pair_right = tf.reshape(index_pair_right, (-1,))

        # Shapes to use for broadcasting the indices.
        batch_size = tf.shape(left_nodes)[0]
        left_indices_shape = tf.stack(
            [batch_size, tf.shape(index_pair_left)[0]], axis=0
        )
        right_indices_shape = tf.stack(
            [batch_size, tf.shape(index_pair_right)[0]], axis=0
        )

        # Collect the features.
        index_pair_left = tf.broadcast_to(index_pair_left, left_indices_shape)
        index_pair_right = tf.broadcast_to(
            index_pair_right, right_indices_shape
        )
        left_features = tf.gather(
            left_nodes, index_pair_left, axis=1, batch_dims=1
        )
        right_features = tf.gather(
            right_nodes, index_pair_right, axis=1, batch_dims=1
        )

        # Combine into a single tensor.
        split_feature_shape = tf.stack(
            (batch_size, num_left_nodes, num_right_nodes, num_features), axis=0
        )
        left_features = tf.reshape(left_features, split_feature_shape)
        right_features = tf.reshape(right_features, split_feature_shape)
        return tf.stack([left_features, right_features], axis=3)


def compute_pairwise_similarities(
    similarity_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    *,
    right_features: tf.Tensor,
    left_features: tf.Tensor,
) -> tf.Tensor:
    """
    Computes some similarity metric between every possible combination of
    two sets of features. This can be thought of as a bipartite graph
    problem, where the two inputs are feature sets for the left and right nodes.

    Args:
        similarity_function: A function that computes a similarity metric
            between two batches of feature vectors. Should return a batch
            of scalar similarity values.
        right_features: The first set of features,
            with shape `[batch_size, n_nodes, n_features]`.
        left_features: The second set of features,
            with shape `[batch_size, n_nodes, n_features]`.

    Returns:
        The similarities computed between every combination of left and
        right features. Will have shape
        `[batch_size, n_left, n_right]`.

    """
    # Compute all possible combinations.
    combinations = compute_bipartite_edge_features(
        left_features, right_features
    )

    combinations_shape = tf.shape(combinations)
    combinations_outer_shape = combinations_shape[:3]
    num_features = combinations_shape[-1]

    # Reshape so we can compute similarities in one pass.
    flat_shape = tf.stack((-1, 2, num_features), axis=0)
    combinations = tf.reshape(combinations, flat_shape)
    tracklet_features_flat = combinations[:, 0, :]
    detection_features_flat = combinations[:, 1, :]

    # Compute similarities.
    similarities = similarity_function(
        tracklet_features_flat, detection_features_flat
    )

    # Add the other dimensions back.
    return tf.reshape(similarities, combinations_outer_shape)


def make_adjacency_matrix(edge_features: tf.Tensor) -> tf.Tensor:
    """
    Computes a dense adjacency matrix based on provided bipartite edge features.
    Zeros will be used for any edge that does not exist.

    Args:
        edge_features: The edge features. Should have the shape
            [batch_size, n_left_nodes, n_right_nodes, n_features].

    Returns:
        The adjacency matrix that it created, which will have the shape
        [batch_size, n_left_nodes + n_right_nodes, n_left_nodes + n_right_nodes,
         n_features]. It will put the left node features first in the indexing.

    """
    features_rank_4 = tf.assert_rank(edge_features, 4)

    with tf.control_dependencies([features_rank_4]):
        edge_feature_shape = tf.shape(edge_features)
        num_left_nodes = edge_feature_shape[1]
        num_right_nodes = edge_feature_shape[2]

        # Pad the unconnected edges with zeros.
        paddings = tf.stack(
            [0, 0, 0, num_right_nodes, num_left_nodes, 0, 0, 0], axis=0
        )
        paddings = tf.reshape(paddings, (4, 2))
        return tf.pad(edge_features, paddings)


def _single_complete_bipartite_adjacency_matrix(
    num_left_nodes: tf.Tensor,
    num_right_nodes: tf.Tensor,
    *,
    adjacency_shape: tf.Tensor,
) -> tf.Tensor:
    """
    Creates the binary adjacency matrix for a complete bipartite graph.

    Args:
        num_left_nodes: The number of nodes on the left side. Should be a
            0D tensor.
        num_right_nodes: The number of nodes on the right side. Should be a
            0D tensor.
        adjacency_shape: The shape to use for the output dense adjacency matrix.
            This can be specified to a shape larger than necessary to facilitate
            batching the output of this function across multiple graphs.

    Returns:
        The binary adjacency matrix that it created, which will have the shape
        `[n_left_nodes + n_right_nodes, n_left_nodes + n_right_nodes]`.

    """
    num_left_nodes = tf.ensure_shape(num_left_nodes, ())
    num_right_nodes = tf.ensure_shape(num_right_nodes, ())

    # Enumerate all possible pairings between left and right.
    x, y = tf.meshgrid(
        tf.range(num_left_nodes, dtype=tf.int64),
        # Left nodes have the lower half of the indices in our adjacency
        # matrix. Right nodes have the upper half.
        tf.range(
            num_left_nodes, num_left_nodes + num_right_nodes, dtype=tf.int64
        ),
        indexing="ij",
    )
    x = tf.reshape(x, (-1,))
    y = tf.reshape(y, (-1,))
    edge_indices = tf.stack([x, y], axis=-1)
    num_edges = tf.shape(edge_indices)[0]

    adjacency_shape = tf.cast(adjacency_shape, tf.int64)
    adjacency_sparse = tf.SparseTensor(
        edge_indices,
        values=tf.ones(tf.expand_dims(num_edges, 0), dtype=tf.float32),
        dense_shape=adjacency_shape,
    )
    dense_adjacency = tf.sparse.to_dense(adjacency_sparse)

    # We've only produced the upper half of the adjacency matrix, so make it
    # symmetric now.
    return dense_adjacency + tf.transpose(dense_adjacency)


def make_complete_bipartite_adjacency_matrices(
    num_left_nodes: tf.Tensor, num_right_nodes: tf.Tensor
) -> tf.Tensor:
    """
    Creates the binary adjacency matrices for a batch of complete bipartite
    graphs.

    Args:
        num_left_nodes: A vector, representing the number of nodes on the
            left side of each graph.
        num_right_nodes: A vector, representing the number of nodes on the
            right side of each graph.

    Returns:
        The binary adjacency matrices that it created, which will have the shape
        `[batch_size, max_n_left_nodes + max_n_right_nodes, max_n_left_nodes +
          max_n_right_nodes]`

    """
    num_left_nodes = tf.convert_to_tensor(num_left_nodes)
    num_right_nodes = tf.convert_to_tensor(num_right_nodes)

    # Our output has to be large enough to hold the largest adjacency matrix.
    max_num_left_nodes = tf.reduce_max(num_left_nodes)
    max_num_right_nodes = tf.reduce_max(num_right_nodes)
    output_shape = tf.stack([max_num_left_nodes + max_num_right_nodes] * 2)

    return tf.map_fn(
        lambda n: _single_complete_bipartite_adjacency_matrix(
            n[0], n[1], adjacency_shape=output_shape
        ),
        (num_left_nodes, num_right_nodes),
        fn_output_signature=tf.TensorSpec([None, None], dtype=tf.float32),
    )


def augment_adjacency_matrix(
    *, adjacency_matrix: tf.Tensor, node_features: tf.Tensor
) -> tf.Tensor:
    """
    Creates an "augmented" adjacency matrix where the corresponding features
    for both nodes in each edge are concatenated to that edge feature. For
    instance, the feature at `(n, m)` is going to be

    `concat(adjacency_matrix[n, m], node_features[n], node_features[m])`.

    Args:
        adjacency_matrix: The input adjacency matrix. Should have the shape
            `[batch_size, n_nodes, n_nodes, n_edge_features]`.
        node_features: The complete node features. Should have the shape
            `[batch_size, n_nodes, n_node_features]`.

    Returns:
        The adjacency matrix concatenated with corresponding node features.

    """
    adjacency_rank_4 = tf.assert_rank(adjacency_matrix, 4)
    nodes_rank_3 = tf.assert_rank(node_features, 3)

    with tf.control_dependencies([adjacency_rank_4, nodes_rank_3]):
        node_features_shape = tf.shape(node_features)
        batch_size = node_features_shape[0]
        num_nodes = node_features_shape[1]
        num_features = node_features_shape[2]

        expanded_feature_shape = tf.stack(
            (batch_size, num_nodes, num_nodes, num_features), axis=0
        )
        node_features_left = tf.broadcast_to(
            tf.expand_dims(node_features, axis=1), expanded_feature_shape
        )
        node_features_right = tf.transpose(node_features_left, (0, 2, 1, 3))

        # Concatenate on the feature dimension.
        return tf.concat(
            (adjacency_matrix, node_features_left, node_features_right), axis=3
        )


def degree_power(
    adjacency: tf.Tensor, power: Union[tf.Tensor, float]
) -> tf.Tensor:
    """
    Computes a degree matrix raised to a power. This has similar functionality
    to the function of the same name in Spektral. Unfortunately, the Spektral
    version does not support tensor inputs.

    Args:
        adjacency: The adjacency matrix. Should have shape
            `[batch_size, num_nodes, num_nodes]`.
        power: The power to raise it to.

    Returns:
        The adjacency matrix raised to this power.

    """
    adjacency_rank_3 = tf.assert_rank(adjacency, 3)

    with tf.control_dependencies([adjacency_rank_3]):
        batch_size = tf.shape(adjacency)[0]

    degrees = tf.pow(tf.reduce_sum(adjacency, axis=2), power)
    degrees = tf.reshape(degrees, (batch_size, -1))
    # Filter infinite values.
    degrees = tf.where(
        tf.math.is_inf(degrees), tf.zeros_like(degrees), degrees
    )

    return tf.linalg.diag(degrees)


def normalized_adjacency(
    adjacency: tf.Tensor, symmetric: bool = True
) -> tf.Tensor:
    """
    Normalizes the given adjacency matrix using the degree matrix as either
    (D^{-1}A) or (D^{-1/2}AD^{-1/2}) (symmetric normalization).
    This has similar functionality to the function of the same name in
    Spektral. Unfortunately, the Spektral version does not support tensor
    inputs.

    Args:
        adjacency: The adjacency matrix to normalize.
        symmetric: Whether to compute the symmetric normalization or not.

    Returns:
        The normalized adjacency matrix.

    """
    if symmetric:
        normalized_degree = degree_power(adjacency, -0.5)
        return tf.matmul(
            tf.matmul(normalized_degree, adjacency), normalized_degree
        )
    else:
        normalized_degree = degree_power(adjacency, -1.0)
        return tf.matmul(normalized_degree, adjacency)


def gcn_filter(adjacency: tf.Tensor, symmetric: bool = True) -> tf.Tensor:
    """
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).

    This has similar functionality to the function of the same name in
    Spektral. Unfortunately, the Spektral version does not support tensor
    inputs.

    Args:
        adjacency: The adjacency matrix. Should have a shape of
            `[batch_size, num_nodes, num_nodes]`.
        symmetric: boolean, whether to normalize the matrix as
            (D^{-frac{1}{2}}AD^{-frac{1}{2}}) or as (D^{-1}A)

    Returns:
        Tensor with rank 2 or 3, same as `adjacency`.

    """
    adjacency_rank_3 = tf.assert_rank(adjacency, 3)
    with tf.control_dependencies([adjacency_rank_3]):
        num_nodes = tf.shape(adjacency)[1]

    # Add self-connections.
    adjacency_hat = adjacency + tf.eye(num_nodes, dtype=adjacency.dtype)

    return normalized_adjacency(adjacency_hat, symmetric=symmetric)


def bound_adjacency(adjacency_matrix: tf.Tensor) -> tf.Tensor:
    """
    Limit an adjacency matrix to have values between 0 and inf.

    Args:
        adjacency_matrix: The raw adjacency matrix, of shape
            `[batch_size, n_nodes, n_nodes, n_features]`.

    Returns:
        The same adjacency matrix, but bounded.

    """
    return tf.square(adjacency_matrix)
