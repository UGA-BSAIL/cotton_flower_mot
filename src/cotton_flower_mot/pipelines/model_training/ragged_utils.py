"""
Utilities for dealing with `RaggedTensor`s.
"""


from typing import Any, Callable, Tuple

import tensorflow as tf


def ragged_map_fn(function: Callable, elements: Any, **kwargs: Any) -> Any:
    """
    `map_fn` doesn't work with RaggedTensors on the GPU due to a bug in TF
    (https://github.com/tensorflow/tensorflow/issues/46635). This
    implementation works around that by operating on a padded version of the
    input and cutting it down. Obviously, this is inefficient, so beware.

    This is specifically designed to work with `RaggedTensors` that look like
    `[batch_size, (row_len), ...]`. `function` will receive tensors of shape
    `[row_len, ...]`, where `row_len` can vary between calls.

    Args:
        function: The function to apply to each row in the `RaggedTensor`.
        elements: The `RaggedTensor`. Only the second dimension should be
            ragged.
        **kwargs: Will be forwarded to `map_fn`.

    Returns:
        The result of calling `map_fn` on the input, as if the op was
        successfully run.

    """
    # Transform ragged inputs to dense.
    def _nested_to_dense(_elements: Any) -> Any:
        if not tf.is_tensor(_elements) and hasattr(_elements, "__getitem__"):
            # We have multiple elements.
            dense_elements = [_nested_to_dense(e) for e in _elements]
            return dense_elements

        elif isinstance(_elements, tf.RaggedTensor):
            # It's a singleton RaggedTensor.
            assert (
                _elements.get_shape()[1] is None
            ), "Second dimension should be ragged."
            return _elements.to_tensor(), _elements.row_lengths()

        # Otherwise, it's a singleton normal tensor and needs no conversion.
        return _elements

    def _clip_dense(_elements: Any, _dense_elements: Any) -> Any:
        if not tf.is_tensor(_elements) and hasattr(_elements, "__getitem__"):
            # We have multiple elements.
            clipped_elements = [
                _clip_dense(e, d) for e, d in zip(_elements, _dense_elements)
            ]
            return clipped_elements

        elif isinstance(_elements, tf.RaggedTensor):
            # We have to clip the row from this RaggedTensor.
            element, length = _dense_elements
            return element[:length]

        # Otherwise, it's a singleton normal tensor and needs no clipping.
        return _dense_elements

    def _trim_and_forward(element: Any) -> Any:
        """
        Trims padding and forwards to the actual function.

        Args:
            element: The input element, with RaggedTensors converted to a
                tuple of the padded tensor and the row length.

        Returns:
            The result of the function application.

        """
        return function(_clip_dense(elements, element))

    dense_input = _nested_to_dense(elements)
    return tf.map_fn(_trim_and_forward, dense_input, **kwargs)
