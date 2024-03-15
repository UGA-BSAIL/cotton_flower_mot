from functools import cached_property
from typing import Tuple

from pydantic import validator
from pydantic.dataclasses import dataclass


@dataclass()
class ModelConfig:
    """
    Encapsulates configuration for the model.

    Attributes:
        image_input_shape: The shape of the detections and tracklets being
            input to the appearance feature extractor (height, width, channels).
        frame_input_shape: The shape of the raw video frames from the dataset
         (height, width, channels).
        detection_model_input_shape: The shape of the input to the detection
            model (height, width, channels).
        raw_yolo_input_shape: The expected input shape of the YOLO model we
            are using. YOLO has specific requirements for input shapes,
            and they might not match the shape of the raw frame inputs.
        rot_net_input_shape: The shape of the input to use for RotNet
            pretraining. Should be square so rotated images can be stored in a
            single tensor. (height, width, channels)
        colorization_input_shape: The shape of the input to use for colorization
            pretraining.
        colorization_output_shape: The shape of the outputs from the
            colorization network. The last dimension is the number of
            histogram bins.
        num_reduction_stages: How many initial reduction stages to add to the
            detector. Every stage will reduce the size of the heatmap output
            by a factor of 2.
        detection_sigma: Sigma to use for detection heatmaps.

        roi_pooling_size: Size to use for the ROI pooling operation when
            downsampling the detector features to appearance features.
        num_appearance_features: The number of appearance features to extract
            from each image.
        num_node_features: Number of output node features to use for the
            CensNet blocks.
        num_edge_features: Number of output edge features to use for the
            CensNet blocks.

        sinkhorn_lambda: The lambda parameter to use for Sinkhorn normalization.

        nominal_detection_size: The average size of a detection, in fractions
            of a frame. This will be used as a baseline when predicting bounding
            box sizes. In the form `(width, height)`.

    """

    image_input_shape: Tuple[int, int, int]
    frame_input_shape: Tuple[int, int, int]
    detection_model_input_shape: Tuple[int, int, int]
    raw_yolo_input_shape: Tuple[int, int, int]
    rot_net_input_shape: Tuple[int, int, int]
    colorization_input_shape: Tuple[int, int, int]
    colorization_output_shape: Tuple[int, int, int]
    num_reduction_stages: int
    detection_sigma: float

    roi_pooling_size: int
    num_appearance_features: int
    num_node_features: int
    num_edge_features: int

    sinkhorn_lambda: float

    nominal_detection_size: Tuple[float, float]

    @validator("rot_net_input_shape")
    def rot_net_input_is_square(
        cls, value: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Ensures that the RotNet input is square.

        Args:
            value: The value to check.

        Returns:
            The same value.

        """
        height, width, _ = value
        if height != width:
            raise ValueError("RotNet input must be square.")
        return value

    @cached_property
    def heatmap_size(self) -> Tuple[int, int]:
        """
        Returns:
            The size of the heatmaps from the detector, in the form
            `(width, height)`.

        """
        down_sample_factor = 2**self.num_reduction_stages
        input_height, input_width, _ = self.detection_model_input_shape
        return (
            input_width // down_sample_factor,
            input_height // down_sample_factor,
        )
