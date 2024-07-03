from functools import partial
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from keras import layers

from ....assignment import (
    do_hard_assignment,
    solve_optimal_transport,
    add_births_and_deaths,
)
from ..ragged_utils import ragged_map_fn


class AssociationLayer(layers.Layer):
    """
    Custom layer that computes association matrix and applies the Sinkhorn
    algorithm.
    """

    def __init__(self, sinkhorn_lambda: float = 10.0, **kwargs: Any):
        """
        Args:
            sinkhorn_lambda: The Sinkhorn lambda value.
            kwargs: Will be forwarded to the superclass.

        """
        super().__init__(**kwargs)

        self._lambda = sinkhorn_lambda

    @staticmethod
    def _compute_row_or_column_sum(
        length: tf.Tensor, *, other_dim_length: tf.Tensor
    ) -> tf.Tensor:
        """
        For Sinkhorn normalization, we expect rows and columns to sum to one,
        except for the births/deaths row/column, which we expect to sum to their
        length of the other dimension.

        Args:
            length: The length of the row or column.
            other_dim_length: The length of the other dimension.

        Returns:
            The expected sums. This will have the form `[1, 1, ..., length]`.

        """
        row_shape = tf.expand_dims(length, axis=0)
        sums = tf.ones(row_shape, dtype=tf.float32)

        # Add the last element.
        last_element = tf.expand_dims(other_dim_length, axis=0)
        return tf.concat((sums, tf.cast(last_element, tf.float32)), axis=0)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
        **_,
    ) -> tf.RaggedTensor:
        """

        Args:
            inputs:
                Has three elements:
                - affinity_scores: The affinity scores computed for each
                    possible tracklet/detection pair. Should have a shape of
                    `[batch_size, max_n_tracklets, max_n_detections]`.
                - num_detections: The number of detections in each example.
                    Should be a vector of shape `[batch_size]`.
                - num_tracklets: The number of tracklets in each example.
                    Should be a vector of shape `[batch_size]`.
            training: Whether the layer should operate in training mode.

        Returns:
            The normalized optimal transport matrix. This will
            be a `RaggedTensor` where the second dimension is ragged,
            so it will have the shape
            `[batch_size, (n_tracklets + 1) * (n_detections + 1)]`.

        """
        affinity_scores, num_detections, num_tracklets = inputs
        one = tf.constant(1, dtype=num_detections.dtype)

        def _normalize(
            element: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        ) -> tf.Tensor:
            affinity_matrix, _num_detections, _num_tracklets = element
            # For the output, will add an extra row and column for
            # births/deaths.
            affinity_shape = tf.shape(affinity_matrix)
            output_flat_length = tf.math.reduce_prod(
                affinity_shape + tf.ones_like(affinity_shape)
            )
            output_flat_length = tf.cast(
                output_flat_length, _num_detections.dtype
            )

            def _pad_and_flatten(association: tf.Tensor) -> tf.Tensor:
                # Flatten.
                transport_flat = tf.reshape(association, (-1,))
                # Re-pad so the outputs all have the same size.
                padding = tf.stack(
                    (
                        0,
                        output_flat_length
                        - (_num_tracklets + one) * (_num_detections + one),
                    )
                )
                return tf.pad(transport_flat, tf.expand_dims(padding, 0))

            # Remove the padding.
            affinity_un_padded = affinity_matrix[
                :_num_tracklets, :_num_detections
            ]
            # Add additional row and column for track births/deaths.
            affinity_expanded = tf.pad(affinity_un_padded, [[0, 1], [0, 1]])

            row_sums = self._compute_row_or_column_sum(
                _num_tracklets, other_dim_length=_num_detections
            )
            column_sums = self._compute_row_or_column_sum(
                _num_detections, other_dim_length=_num_tracklets
            )

            # Add fake batch dimension.
            affinity_expanded = tf.expand_dims(affinity_expanded, axis=0)
            row_sums = tf.expand_dims(row_sums, axis=0)
            column_sums = tf.expand_dims(column_sums, axis=0)

            # Normalize it.
            transport, _ = solve_optimal_transport(
                # Cost matrix is -affinity.
                -affinity_expanded,
                lamb=self._lambda,
                row_sums=row_sums,
                column_sums=column_sums,
            )
            # Remove extraneous batch dimension.
            transport = transport[0]
            return _pad_and_flatten(transport)

        # Unfortunately, we can't have padding for the affinity scores, because
        # it affects the optimization. Therefore, this process has to be done
        # with map_fn instead of vectorized.
        sinkhorn_dense = tf.map_fn(
            _normalize,
            (affinity_scores, num_detections, num_tracklets),
            fn_output_signature=(
                tf.TensorSpec(shape=[None], dtype=tf.float32)
            ),
        )

        # Convert to a ragged tensor.
        row_lengths = (num_detections + one) * (num_tracklets + one)
        to_ragged = partial(tf.RaggedTensor.from_tensor, lengths=row_lengths)
        return to_ragged(sinkhorn_dense)

    def get_config(self) -> Dict[str, Any]:
        return dict(sinkhorn_lambda=self._lambda)


class HungarianLayer(layers.Layer):
    """
    Custom layer that applies the Hungarian algorithm to the input to compute a
    final hard assignment matrix.
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        """
        Args:
            threshold: The threshold to apply to the Sinkhorn matrix before
                applying the Hungarian algorithm.
            kwargs: Will be forwarded to the superclass.

        """
        super().__init__(**kwargs)

        self._threshold = threshold

    def call(
        self,
        inputs: Tuple[tf.RaggedTensor, tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
        **_,
    ) -> tf.RaggedTensor:
        """

        Args:
            inputs:
                - sinkhorn: The flattened Sinkhorn matrix. Should have a
                    shape of
                    `[batch_size, (n_tracklets + 1) * (n_detections + 1)]`.
                - num_detections: The number of detections in each example.
                    Should be a vector of shape `[batch_size]`.
                - num_tracklets: The number of tracklets in each example.
                    Should be a vector of shape `[batch_size]`.
            training: Whether the layer should operate in training mode.

        Returns:
            The hard assignment matrix. This will be a `RaggedTensor` where
            the second dimension is ragged, so it will have the shape
            `[batch_size, (n_tracklets + 1) * (n_detections + 1)]`.

        """
        sinkhorn, num_detections, num_tracklets = inputs
        one = tf.constant(1, dtype=num_detections.dtype)

        def _hungarian(
            element: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            _sinkhorn, _num_detections, _num_tracklets = element
            # Un-flatten the sinkhorn matrix.
            _sinkhorn = tf.reshape(
                _sinkhorn,
                (
                    _num_tracklets + one,
                    _num_detections + one,
                ),
            )

            assignment = do_hard_assignment(
                _sinkhorn, threshold=self._threshold
            )
            assignment = add_births_and_deaths(assignment)

            return tf.reshape(assignment, (-1,))

        return ragged_map_fn(
            _hungarian,
            (sinkhorn, num_detections, num_tracklets),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=0, dtype=tf.bool
            ),
        )
