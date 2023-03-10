"""
Tests for the `combined_model` module.
"""


import numpy as np
import pytest
import tensorflow as tf
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import combined_model
from src.cotton_flower_mot.pipelines.model_training.centernet_model import (
    build_detection_model,
)
from src.cotton_flower_mot.pipelines.model_training.gcnn_model import (
    build_tracking_model,
)


@pytest.mark.integration
@pytest.mark.slow
def test_build_combined_model_smoke(faker: Faker) -> None:
    """
    Tests that we can build and use the combined model.

    Args:
        faker: The fixture to use for generating fake data.
        is_training: If true, test the training configuration of the model.
            Otherwise, test the inference configuration.

    """
    # Arrange.
    detection_shape = (540, 960, 3)

    # Create fake input data.
    batch_size = faker.random_int(max=4)
    current_frames = faker.tensor((batch_size,) + detection_shape)
    previous_frames = faker.tensor((batch_size,) + detection_shape)
    detection_geometry = faker.bounding_boxes(batch_size=batch_size)
    tracklet_geometry = faker.bounding_boxes(batch_size=batch_size)

    config = faker.model_config(detection_input_shape=detection_shape)

    # Act.
    feature_extractor, detection_model = build_detection_model(config)
    tracking_model = build_tracking_model(
        config, feature_extractor=feature_extractor
    )
    model = combined_model.build_combined_model(
        config, detector=detection_model, tracker=tracking_model
    )

    # Test the model.
    model_inputs = [
        current_frames,
        previous_frames,
        detection_geometry,
        tracklet_geometry,
    ]
    heatmap, dense_geometry, bboxes, sinkhorn, assignment = model(model_inputs)

    # Assert.
    heatmap_shape = tf.shape(heatmap).numpy()
    dense_geometry_shape = tf.shape(dense_geometry).numpy()

    # Check that detections outputs are sized as expected.
    np.testing.assert_array_equal(heatmap_shape[:3], dense_geometry_shape[:3])
    # We should have a single-channel heatmap.
    assert heatmap_shape[-1] == 1
    # We should have four geometry channels.
    assert dense_geometry_shape[-1] == 4

    # Bounding boxes should be ragged.
    assert type(bboxes) == tf.RaggedTensor

    # Make sure the association matrices are the expected size.
    row_sizes = tracklet_geometry.row_lengths().numpy()
    col_sizes = detection_geometry.row_lengths().numpy()
    expected_lengths = row_sizes * col_sizes
    np.testing.assert_array_equal(
        sinkhorn.row_lengths().numpy(), expected_lengths
    )
    np.testing.assert_array_equal(
        assignment.row_lengths().numpy(), expected_lengths
    )
