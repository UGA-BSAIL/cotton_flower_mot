"""
Utilities for losses and metrics.
"""

from typing import Tuple, Union

import tensorflow as tf

MaybeRagged = Union[tf.Tensor, tf.RaggedTensor]
"""
Represents something that might be a RaggedTensor.
"""


def correct_ragged_mismatch(
    y_true: MaybeRagged, y_pred: MaybeRagged
) -> Tuple[MaybeRagged, MaybeRagged]:
    """
    Workaround for a bug in TF 2.4 where we can't input `y_true` as a
    ragged tensor. However, `y_pred` can still be ragged, so in this
    situation, we convert `y_true` to a ragged tensor based on the row
    lengths from `y_pred`.

    Args:
        y_true: The ground-truth labels.
        y_pred: The predicted labels.

    Returns:
        The same true and predicted labels. In the case where `y_pred` is
        ragged and `y_true` is not, `y_true` will be made ragged. Otherwise,
        both will be returned unchanged.


    """
    if isinstance(y_pred, tf.RaggedTensor) and isinstance(y_true, tf.Tensor):
        y_true = tf.RaggedTensor.from_tensor(
            y_true, lengths=y_pred.row_lengths()
        )

    return y_true, y_pred
