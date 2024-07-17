"""
Wrapper that provides a unified API to access models.
"""


import abc
from typing import Tuple, Optional, Dict, Any

import numpy as np

import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf


class DetectionModel(abc.ABC):
    """
    Wrapper that provides a unified API to access detection models.
    """

    @abc.abstractmethod
    def detect(self, frame: np.array) -> Tuple[np.array, np.array]:
        """
        Runs the detection model on the specified frame.

        Args:
            frame: The frame to run detection on.

        Returns:
            - The detection geometry, with confidence.
            - The corresponding appearance features.

        """


class TrackingModel(abc.ABC):
    """
    Wrapper that provides a unified API to access tracking models.
    """

    @abc.abstractmethod
    def track(
        self,
        *,
        detections: np.array,
        detections_appearance: np.array,
        tracklets: np.array,
        tracklets_appearance: np.array,
    ) -> np.array:
        """
        Runs the tracking model on the specified frame.

        Args:
            detections: The detection geometry, without confidence.
            detections_appearance: The corresponding detection appearance
                features.
            tracklets: The tracklet geometry, without confidence.
            tracklets_appearance: The corresponding tracklet appearance
                features.

        Returns:
            The predicted Sinkhorn matrix.

        """


class _RemoteModelMixin:
    """
    Mixin class for models that use gRPC to run remotely.
    """

    def __init__(
        self,
        *,
        endpoint: str = "localhost:8500",
        model_name: str,
        version: Optional[int] = None,
        signature_name: str = "serving_default",
    ):
        """
        Args:
            endpoint: The gRPC endpoint to connect to.
            model_name: The name of the model to run.
            version: The version of the model to use. If not specified,
                it will use the default version.
            signature_name: The signature name to use for inference.

        """
        self.__channel = grpc.insecure_channel(endpoint)
        self.__stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.__channel
        )
        self.__model_name = model_name
        self.__version = version
        self.__signature_name = signature_name

    def _predict_grpc(
        self, input_dict: Dict[str, np.array]
    ) -> Dict[str, np.array]:
        """
        Creates and runs a gRPC prediction request.

        Args:
            input_dict: The dictionary of inputs, mapping input names to
                input data.

        Returns:
            The dictionary of outputs, mapping output names to output data.

        """
        # Create a gRPC request made for prediction
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.__model_name
        request.model_spec.signature_name = self.__signature_name
        if self.__version is not None:
            # Use a specific version.
            request.model_spec.version.value = self.__version

        # Set the input as the data
        for input_name, input_data in input_dict.items():
            request.inputs[input_name].CopyFrom(
                tf.make_tensor_proto(input_data)
            )

        # Send the gRPC request to the TF Server
        result = self.__stub.Predict(request)
        return {k: tf.make_ndarray(v) for k, v in result.outputs.items()}


class RemoteDetectionModel(_RemoteModelMixin, DetectionModel):
    """
    Wrapper that provides a unified API to access remote detection models.
    """

    def __init__(
        self,
        detections_output: str = "input.to_tensor_1",
        appearance_output: str = "input.to_tensor_2",
        **kwargs: Any,
    ):
        """
        Args:
            detections_output: The name of the output for the detections.
            appearance_output: The name of the output for the appearance.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(**kwargs)

        self.__detections_output = detections_output
        self.__appearance_output = appearance_output

    def detect(self, frame: np.array) -> Tuple[np.array, np.array]:
        outputs = self._predict_grpc(
            {"detections_frame": frame[None, :, :, :].astype(np.float32)}
        )
        return (
            outputs[self.__detections_output][0],
            outputs[self.__appearance_output][0],
        )


class RemoteTrackingModel(_RemoteModelMixin, TrackingModel):
    """
    Wrapper that provides a unified API to access remote tracking models.
    """

    def track(
        self,
        *,
        detections: np.array,
        detections_appearance: np.array,
        tracklets: np.array,
        tracklets_appearance: np.array,
    ) -> np.array:
        num_detections = np.array([[detections.shape[0]]], dtype=np.int32)
        num_tracklets = np.array([[tracklets.shape[0]]], dtype=np.int32)
        input_dict = {
            "detection_appearance_flat": detections_appearance[None, :, :],
            "detection_appearance_row_lengths": num_detections,
            "tracklet_appearance_flat": tracklets_appearance[None, :, :],
            "tracklet_appearance_row_lengths": num_tracklets,
            "detection_geometry_flat": detections[None, :, :],
            "detection_geometry_row_lengths": num_detections,
            "tracklet_geometry_flat": tracklets[None, :, :],
            "tracklet_geometry_row_lengths": num_tracklets,
        }

        outputs = self._predict_grpc(input_dict)
        return outputs["input.to_tensor"][0]
