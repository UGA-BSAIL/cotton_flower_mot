"""
Message-passing-based implementation of CensNet.
"""

from spektral.layers import MessagePassing
import tensorflow as tf
from typing import Tuple, Any, TypeVar, Dict
import keras

MaybeSparse = TypeVar("MaybeSparse", tf.Tensor, tf.SparseTensor)
"""
Used to type parameters that can be either dense or sparse tensors.
"""


class CensNet(MessagePassing):
    """
    Message-passing-based implementation of CensNet.
    """

    def __init__(
        self,
        num_nodes_out: int,
        num_edges_out: int,
        activation: str = "relu",
        use_bias: bool = True,
        **kwargs: Any
    ):
        """
        Args:
            num_nodes_out: The number of node output features.
            num_edges_out: The number of edge output features.
            activation: The activation function to use.
            use_bias: Whether to use biases as well.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(**kwargs)

        self.num_nodes_out = num_nodes_out
        self.num_edges_out = num_edges_out
        self._activation_name = activation
        self._activation = keras.activations.get(activation)
        self._use_bias = use_bias

        self.node_weights = None
        self.edge_weights = None
        self.edge_weight_vector = None
        self.node_weight_vector = None
        self.node_bias = None
        self.edge_bias = None

    def get_config(self) -> Dict[str, Any]:
        return dict(
            num_nodes_out=self.num_nodes_out,
            num_edges_out=self.num_edges_out,
            activation=self._activation_name,
            **super().get_config(),
        )

    @staticmethod
    def _triangular_adjacency(adjacency: MaybeSparse) -> MaybeSparse:
        """
        Gets the triangular version of the adjacency matrix, removing redundant
        values.

        Args:
            adjacency: The full adjacency matrix, with shape
                (n_nodes, n_nodes).

        :return: The upper triangle of the adjacency matrix, with the lower
            triangle set to zero.
        """
        if isinstance(adjacency, tf.Tensor):
            return tf.linalg.band_part(adjacency, 0, -1)

        elif isinstance(adjacency, tf.SparseTensor):
            # Mask out the bottom triangle.
            mask = adjacency.indices[:, 0] <= adjacency.indices[:, 1]
            return tf.sparse.retain(adjacency, mask)

    @classmethod
    def incidence_matrix(
        cls,
        adjacency: MaybeSparse,
    ) -> MaybeSparse:
        """
        Creates the corresponding incidence matrix for a graph with a particular
        adjacency matrix.

        Args:
            adjacency: The binary adjacency matrix. Should have shape
                (n_nodes, n_nodes), and be symmetric.

        Returns:
            The computed incidence matrix. It will have a shape of
            (n_nodes, n_edges).

        """
        # Convert to upper triangular.
        adjacency = cls._triangular_adjacency(adjacency)

        # The adjacency matrix should be sparse, so get the indices of the
        # edges.
        if isinstance(adjacency, tf.SparseTensor):
            connected_node_indices = adjacency.indices
        else:
            connected_node_indices = tf.where(adjacency)

        # Match each edge with one of the nodes connected by that edge. We refer
        # to the two nodes connected by each edge as "right" and "left",
        # for convenience.
        edge_indices = tf.range(
            tf.shape(connected_node_indices)[0], dtype=tf.int64
        )
        edges_with_left_nodes = tf.stack(
            [connected_node_indices[:, 0], edge_indices], axis=1
        )
        edges_with_right_nodes = tf.stack(
            [connected_node_indices[:, 1], edge_indices], axis=1
        )

        # We now have all the points that should go in the sparse incidence
        # matrix.
        edge_indicators = tf.ones_like(edge_indices, dtype=tf.float32)
        num_nodes = tf.cast(tf.shape(adjacency)[0], tf.int64)
        num_edges = tf.cast(tf.shape(connected_node_indices)[0], tf.int64)
        output_shape = tf.stack([num_nodes, num_edges])
        left_sparse = tf.SparseTensor(
            indices=edges_with_left_nodes,
            values=edge_indicators,
            dense_shape=output_shape,
        )
        left_sparse = tf.sparse.reorder(left_sparse)
        right_sparse = tf.SparseTensor(
            indices=edges_with_right_nodes,
            values=edge_indicators,
            dense_shape=output_shape,
        )
        right_sparse = tf.sparse.reorder(right_sparse)
        # Combine the matrices for the left and right nodes.
        combined_sparse = tf.sparse.maximum(left_sparse, right_sparse)

        if isinstance(adjacency, tf.SparseTensor):
            return combined_sparse
        else:
            return tf.sparse.to_dense(combined_sparse)

    @classmethod
    def _square_incidence(
        cls, incidence: tf.SparseTensor, edge_weights: tf.Tensor | None = None
    ) -> tf.SparseTensor:
        """
        A special, optimized method that computes the square (M@M') of a sparse
        incidence matrix.

        Args:
            incidence: The incidence matrix, of shape [V, E]
            edge_weights: A vector of weights, with a length of E. If
                provided, will compute `M @ diag(edge_weights) @ M'`.

        Returns:
            The square of the matrix.

        """
        # To compute the diagonal, we just need to sum along the rows.
        diagonal_weighted_incidence = incidence
        if edge_weights is not None:
            edge_weights = tf.squeeze(edge_weights, axis=-1)
            diagonal_weighted_incidence = incidence * edge_weights
        row_sums = tf.sparse.reduce_sum(diagonal_weighted_incidence, axis=1)
        diag_indices = tf.range(0, tf.shape(row_sums)[0], dtype=tf.int64)
        diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
        diag_values = row_sums

        # Computing the off-diagonals is a little trickier. Basically,
        # we check the indices to see if there are any cases where the row
        # index is different, but the column index is the same. This would
        # indicate that when we take the dot product of those two rows (as we
        # would when performing the matrix multiplication naively),
        # the result will be a one, which tells us the position of a one in
        # the output.
        row_indices = incidence.indices[:, 0]
        col_indices = incidence.indices[:, 1]
        different_rows = row_indices[None, :] != row_indices[:, None]
        same_cols = col_indices[None, :] == col_indices[:, None]
        ones_mask = tf.logical_and(different_rows, same_cols)

        match_indices = tf.where(ones_mask)
        ones_row_indices = tf.gather(row_indices, match_indices[:, 0])
        ones_col_indices = tf.gather(row_indices, match_indices[:, 1])
        ones_indices = tf.stack([ones_row_indices, ones_col_indices], axis=1)

        # Handle weighting if necessary.
        off_diag_values = tf.ones_like(ones_row_indices, dtype=incidence.dtype)
        if edge_weights is not None:
            # The indices for the weights we use will be the matching columns.
            weight_indices = tf.gather(col_indices, match_indices[:, 0])
            off_diag_values = tf.gather(edge_weights, weight_indices)

        # Set these values to one and combine with the diagonal values.
        all_indices = tf.concat([diag_indices, ones_indices], axis=0)
        all_values = tf.concat([diag_values, off_diag_values], axis=0)
        output_shape = tf.stack([incidence.dense_shape[0]] * 2)
        return tf.sparse.reorder(
            tf.SparseTensor(
                indices=all_indices,
                values=all_values,
                dense_shape=output_shape,
            )
        )

    @classmethod
    def line_graph(cls, incidence: tf.SparseTensor) -> tf.SparseTensor:
        """
        Creates the corresponding normal adjacency matrix.

        Args:
            incidence: The binary incidence matrix. Should have shape
                (n_nodes, n_edges).

        Returns:
            The computed line graph adjacency matrix. It will have a shape
            of (n_edges, n_edges) and self-loops.
        """
        incidence_sq = cls._square_incidence(tf.sparse.transpose(incidence))

        num_rows = incidence_sq.dense_shape[0]
        identity = tf.sparse.eye(num_rows, dtype=incidence.dtype)
        edge_adjacency = tf.sparse.add(
            incidence_sq, identity * tf.constant(-2, dtype=incidence.dtype)
        )
        return cls.add_self_loops(edge_adjacency)

    @classmethod
    def add_self_loops(cls, adjacency: MaybeSparse) -> MaybeSparse:
        """
        Adds self-loops to an adjacency matrix if they are not present.

        Args:
            adjacency: The adjacency matrix to add self-loops to.

        Returns:
            The modified matrix.

        """
        if isinstance(adjacency, tf.SparseTensor):
            eye = tf.sparse.eye(
                adjacency.dense_shape[0], dtype=adjacency.dtype
            )
            return tf.sparse.maximum(adjacency, eye)
        else:
            eye = tf.eye(tf.shape(adjacency)[0], dtype=adjacency.dtype)
            return tf.maximum(adjacency, eye)

    @classmethod
    def laplacian(cls, adjacency: tf.SparseTensor) -> tf.SparseTensor:
        """
        Computes the normalized Laplacian matrix as required by GCN:

        `D^(-1/2)@A@D^(-1/2)`, where `A` is the adjacency matrix and `D` is the
        degree matrix.

        Args:
            adjacency: The adjacency matrix, with self-loops.

        Returns:
            The laplacian.

        """
        index_sources = adjacency.indices[:, 0]
        index_targets = adjacency.indices[:, 1]

        # Compute the degree of each node in the adjacency matrix.
        degrees = tf.math.unsorted_segment_sum(
            tf.ones_like(index_sources),
            index_sources,
            adjacency.dense_shape[0],
        )
        # Figure out the degree of the node at each end of each edge.
        source_edge_degrees = tf.gather(degrees, index_sources)
        target_edge_degrees = tf.gather(degrees, index_targets)

        # Compute laplacian weights.
        weights = 1 / tf.sqrt(
            tf.cast(source_edge_degrees * target_edge_degrees, tf.float32)
        )
        return tf.sparse.SparseTensor(
            indices=adjacency.indices,
            values=weights,
            dense_shape=adjacency.dense_shape,
        )

    @classmethod
    def preprocess(
        cls, adjacency: MaybeSparse
    ) -> Tuple[tf.SparseTensor, tf.SparseTensor, tf.SparseTensor]:
        """
        Pre-processes the adjacency matrix such that it can be used in this
        layer.

        Args:
            adjacency: The node adjacency matrix, of shape
                `[n_nodes, n_nodes]`.

        Returns:
            The node adjacency, edge adjacency, and incidence matrix,
            all in the form expected by the layer.

        """
        if not isinstance(adjacency, tf.SparseTensor):
            adjacency = tf.sparse.from_dense(adjacency)

        # Get the incidence matrix.
        incidence = cls.incidence_matrix(adjacency)
        edge_adjacency = cls.line_graph(incidence)

        # Compute normalized laplacian.
        adjacency = cls.add_self_loops(adjacency)
        edge_adjacency = cls.add_self_loops(edge_adjacency)
        laplacian = cls.laplacian(adjacency)
        edge_laplacian = cls.laplacian(edge_adjacency)

        return (
            laplacian,
            edge_laplacian,
            incidence,
        )

    def build(
        self, input_shape: Tuple[Tuple, Tuple[Tuple, Tuple, Tuple], Tuple]
    ) -> None:
        """
        Initializes the layer weights.

        Args:
            input_shape: The input shapes of the node, node adjacency,
                edge adjacency, incidence, and edge feature matrices.

        """
        node_feature_shape, _, edge_feature_shape = input_shape
        nodes_in = node_feature_shape[-1]
        edges_in = edge_feature_shape[-1]

        # Initialize the weights.
        self.node_weights = self.add_weight(
            name="node_weights",
            shape=(nodes_in, self.num_nodes_out),
        )
        self.edge_weights = self.add_weight(
            name="edge_weights",
            shape=(edges_in, self.num_edges_out),
        )
        # Edge weight vector for node update.
        self.edge_weight_vector = self.add_weight(
            name="edge_weight_vector",
            shape=(edges_in,),
        )
        # Node weight vector for edge update.
        self.node_weight_vector = self.add_weight(
            name="node_weight_vector",
            shape=(nodes_in,),
        )

        if self._use_bias:
            self.node_bias = self.add_weight(
                name="node_bias",
                shape=(self.num_nodes_out,),
            )
            self.edge_bias = self.add_weight(
                name="edge_bias",
                shape=(self.num_edges_out,),
            )

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[
            tf.Tensor,
            Tuple[tf.SparseTensor, tf.SparseTensor, tf.SparseTensor],
            tf.Tensor,
        ],
        **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the message passing.

        Args:
            inputs: The inputs to the layer:
                - The node feature matrix, of shape `[n_nodes, node_dims]`
                    - The node adjacency matrix, of shape `[n_nodes, n_nodes]`
                    - The edge adjacency matrix, of shape `[n_edges, n_edges]`
                        This can be computed from the node adjacency matrix with
                        `CensNet.line_graph()`.
                    - The incidence matrix, of shape `[n_nodes, n_edges]`. Can
                        be computed from the node adjacency matrix with
                - The edge feature matrix, of shape `[n_edges, edge_dims]`
                    `CensNet.incidence_matrix()`.
            **kwargs: Will be forwarded to `propagate()`.

        Returns:
            - The new node features, of shape `[n_nodes, n_node_features_out]`.
            - The new edge features, of shape `[n_edges, n_edge_features_out]`.

        """
        (
            node_features,
            (node_adjacency, edge_adjacency, incidence),
            edge_features,
        ) = inputs

        # Pre-compute the adjacency weights.
        edge_weights = tf.matmul(
            edge_features, self.edge_weight_vector[:, None]
        )
        node_weights = tf.matmul(
            node_features, self.node_weight_vector[:, None]
        )
        # Project to the correct dimensions.
        node_update_adjacency_weights = self._square_incidence(
            incidence, edge_weights=edge_weights
        )
        edge_update_adjacency_weights = self._square_incidence(
            tf.sparse.transpose(incidence), edge_weights=node_weights
        )

        # Pre-compute the weighted node and edge features.
        node_features = tf.matmul(node_features, self.node_weights)
        edge_features = tf.matmul(edge_features, self.edge_weights)
        if self._use_bias:
            node_features += self.node_bias
            edge_features += self.edge_bias

        # We're going to propagate twice: Once for the node update, and once
        # for the edge update.
        new_node_features = self.propagate(
            x=node_features,
            a=node_adjacency,
            node_update_adjacency_weights=node_update_adjacency_weights,
            edge_update_adjacency_weights=None,
            **kwargs,
        )
        new_edge_features = self.propagate(
            x=edge_features,
            a=edge_adjacency,
            node_update_adjacency_weights=None,
            edge_update_adjacency_weights=edge_update_adjacency_weights,
            **kwargs,
        )
        return new_node_features, new_edge_features

    def message(
        self,
        x: tf.Tensor,
        *,
        a: tf.SparseTensor | None = None,
        node_update_adjacency_weights: tf.SparseTensor | None = None,
        edge_update_adjacency_weights: tf.SparseTensor | None = None,
        **kwargs: Any
    ) -> tf.Tensor:
        """
        Computes the messages for each node or edge, depending on whether
        we're running the node or edge sub-layer.

        Args:
            x: The node features.
            a: The adjacency matrix.
            node_update_adjacency_weights: The adjacency weights for the node
                update, if we are doing that with the same shape as the `a`.
            edge_update_adjacency_weights: The adjacency weights for the edge
                update, if we are doing that, with the same shape as `a`.
            **kwargs: Will be forwarded to the superclass.

        Returns:
            The messages for each edge.

        """
        if edge_update_adjacency_weights is None:
            # We're processing the nodes.
            messages = self.get_sources(x)
            # This is making the assumption that both the adjacency matrix and
            # `node_update_adjacency_weights` are in canonical row-major order.
            weights = node_update_adjacency_weights.values
        else:
            # We're processing the edges.
            messages = self.get_sources(x)
            weights = edge_update_adjacency_weights.values

        return weights[:, None] * a.values[:, None] * messages

    def update(self, embeddings: tf.Tensor, **_: Any) -> tf.Tensor:
        # Apply the activation.
        return self._activation(embeddings)
