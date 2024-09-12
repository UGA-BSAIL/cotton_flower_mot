"""
Contains custom `Faker` providers.
"""

from typing import Any, Iterable, Optional, Reversible, Tuple

import numpy as np
import tensorflow as tf
from faker import Faker
from faker.providers import BaseProvider

from src.cotton_flower_mot.pipelines.model_training.gcnn_model import (
    ModelConfig,
)


class TensorProvider(BaseProvider):
    """
    Provider for creating random Tensors.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.__faker = Faker()

    def tensor(
        self,
        shape: Reversible[int],
        min_value: float = -100.0,
        max_value: float = 100.0,
    ) -> tf.Tensor:
        """
        Creates a fake tensor with arbitrary values and the given shape.

        Args:
            shape: The shape that the tensor should have.
            min_value: Minimum possible value to include in the tensor.
            max_value: Maximum possible value to include in the tensor.

        Returns:
            The tensor that it created.

        """
        # Build up the tensor from the last dimension inward.
        reverse_dimensions = list(reversed(shape))

        tensor = tf.linspace(
            min_value,
            self.__faker.pyfloat(min_value=min_value, max_value=max_value),
            reverse_dimensions[0],
        )
        for dim_size in reverse_dimensions[1:]:
            tensor = tf.linspace(
                tensor,
                tf.zeros_like(tensor)
                + min_value
                + self.__faker.pyfloat(
                    min_value=min_value, max_value=max_value
                ),
                dim_size,
            )

        return tensor

    def ragged_tensor(
        self,
        *,
        row_lengths: Iterable[int],
        inner_shape: Iterable[int] = (1,),
        **kwargs: Any
    ) -> tf.RaggedTensor:
        """
        Creates a fake `RaggedTensor` with arbitrary values and the given shape.

        Args:
            row_lengths: The lengths to use for each row in the tensor.
            inner_shape: The fixed shape to use for each inner element.
            kwargs: Will be forwarded to `self.tensor`.

        Returns:
            The `RaggedTensor` that it created. The final bounding shape will be
            `[len(row_lengths), max(*row_lengths), *inner_shape]`, where the
            second dimension is ragged.

        """
        # Create the tensor elements.
        num_elements = np.sum(row_lengths)
        elements = self.tensor((num_elements,) + tuple(inner_shape), **kwargs)

        # Convert to a `RaggedTensor`.
        return tf.RaggedTensor.from_row_lengths(elements, row_lengths)

    def detected_objects(
        self,
        image_shape: Tuple[int, ...] = (100, 100, 3),
        batch_size: Optional[int] = None,
    ) -> tf.RaggedTensor:
        """
        Creates a fake set of object detections.

        Args:
            image_shape: The shape to use for each object detection, in the
                form `[height, width, channels]`.
            batch_size: The batch size to use. If not specified, it will be
                chosen randomly.

        Returns:
            The fake object detection crops that it created. It will have the
            shape `[batch_size, n_detections, height, width, channels]`.

        """
        if batch_size is None:
            batch_size = self.random_int(min=1, max=16)

        row_lengths_detections = [
            self.random_int(max=8) for _ in range(batch_size)
        ]
        detections = self.ragged_tensor(
            row_lengths=row_lengths_detections, inner_shape=image_shape
        )

        # Convert to integers to simulate how actual images are.
        return tf.cast(detections, tf.uint8)

    def bounding_boxes(
        self, batch_size: Optional[int] = None
    ) -> tf.RaggedTensor:
        """
        Creates a fake set of bounding boxes.

        Args:
            batch_size: The batch size to use. If not specified, it will be
                chosen randomly.

        Returns:
            The fake bounding boxes that it created, in normalized
            coordinates. It will have the shape `[batch_size, n_detections, 4]`,
            where the last dimension is arranged `[x, y, width, height]`. The
            second dimension will be ragged.

        """
        if batch_size is None:
            batch_size = self.random_int(min=1, max=16)

        row_lengths_bboxes = [
            self.random_int(max=8) for _ in range(batch_size)
        ]
        bboxes = self.ragged_tensor(
            row_lengths=row_lengths_bboxes,
            inner_shape=(4,),
            min_value=0.0,
            max_value=1.0,
        )

        return bboxes

    def model_config(
        self,
        image_shape: Optional[Tuple[int, int, int]] = None,
        detection_input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> ModelConfig:
        """
        Creates fake model configurations.

        Args:
            image_shape: The image shape to use.
            detection_input_shape: The shape to use for the detection model
                input.

        Returns:
            The configuration that it created.

        """
        if image_shape is None and detection_input_shape is None:
            raise ValueError(
                "Either 'image_shape' or 'detection_input_shape' must be "
                "specified."
            )

        detection_shape_multiple = self.random_int(min=1, max=10)
        if detection_input_shape is None:
            # Our detection input shape will be some multiple of the input
            # shape.
            image_height, image_width, image_channels = image_shape
            detection_input_shape = (
                image_height * detection_shape_multiple,
                image_width * detection_shape_multiple,
                image_channels,
            )
        elif image_shape is None:
            # Our image shape will be a fraction of the detection input shape.
            (
                detection_height,
                detection_width,
                detection_channels,
            ) = detection_input_shape
            image_shape = (
                detection_height // detection_shape_multiple,
                detection_width // detection_shape_multiple,
                detection_channels,
            )

        # Same with our frame shame.
        frame_shape_multiple = self.random_int(min=1, max=3)
        input_height, input_width, input_channels = detection_input_shape
        frame_shape = (
            input_height * frame_shape_multiple,
            input_width * frame_shape_multiple,
            input_channels,
        )

        # RotNet input has to be square.
        rot_net_side_length = self.random_int(min=100, max=1000)
        rot_net_shape = (rot_net_side_length, rot_net_side_length, 3)

        return ModelConfig(
            image_input_shape=image_shape,
            detection_model_input_shape=detection_input_shape,
            raw_yolo_input_shape=detection_input_shape,
            rot_net_input_shape=rot_net_shape,
            colorization_input_shape=detection_input_shape,
            colorization_output_shape=detection_input_shape,
            frame_input_shape=frame_shape,
            num_appearance_features=self.random_int(min=1, max=256),
            num_node_features=self.random_int(min=1, max=256),
            num_edge_features=self.random_int(min=1, max=256),
            sinkhorn_lambda=self.__faker.pyfloat(min_value=1, max_value=1000),
            num_reduction_stages=self.random_element([2, 3, 4]),
            detection_sigma=self.random_element([1, 3, 5]),
            nominal_detection_size=(
                self.__faker.pyfloat(min_value=0, max_value=1),
                self.__faker.pyfloat(min_value=0, max_value=1),
            ),
            roi_pooling_size=self.random_int(min=1, max=20),
        )
