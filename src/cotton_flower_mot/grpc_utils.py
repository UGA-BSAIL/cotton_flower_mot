"""
Shared utilities for calling a remote model with gRPC.
"""


from typing import Dict

import numpy as np

from tensorflow_serving.apis import predict_pb2
import tensorflow as tf


def make_predict_request(
    input_dict: Dict[str, np.array], *, model_name: str
) -> predict_pb2.PredictRequest:
    """
    Creates a prediction request message.

    Args:
        input_dict: The dictionary of inputs, mapping input names to
                input data.
        model_name: The name of the model.

    Returns:
        The prediction request message.

    """
    request = predict_pb2.PredictRequest()

    request.model_spec.name = model_name

    # Set the input as the data
    for input_name, input_data in input_dict.items():
        request.inputs[input_name].CopyFrom(
            tf.make_tensor_proto(input_data)
        )

    return request
