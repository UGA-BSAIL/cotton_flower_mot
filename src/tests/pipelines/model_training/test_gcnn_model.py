"""
Tests for the `gcnn_model` module.
"""


from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import gcnn_model
from src.cotton_flower_mot.pipelines.model_training.layers import CUSTOM_LAYERS


@pytest.mark.integration
@pytest.mark.slow
def test_extract_appearance_features_smoke(faker: Faker) -> None:
    """
    Tests that `extract_appearance_features` builds a functional graph.

    Args:
        faker: The fixture for creating fake data.

    """
    # Arrange.
    # Create fake detections and tracklets.
    image_shape = (100, 100, 3)
    batch_size = faker.random_int(min=1, max=16)
    detections = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )
    tracklets = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )

    config = faker.model_config(image_shape=image_shape)

    # Act.
    input_shape = (None,) + image_shape
    detection_input = tf.keras.Input(input_shape, ragged=True)
    tracklet_input = tf.keras.Input(input_shape, ragged=True)

    (
        detection_features,
        tracklet_features,
    ) = gcnn_model.extract_appearance_features(
        detections=detection_input, tracklets=tracklet_input, config=config
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[detection_input, tracklet_input],
        outputs=[detection_features, tracklet_features],
    )

    # Apply the model to the inputs we generated.
    got_detection_features, got_tracklet_features = model.predict(
        (detections, tracklets)
    )

    # Assert.
    # Make sure that our results are the right shape.
    detection_shape = got_detection_features.bounding_shape().numpy()
    tracklet_shape = got_tracklet_features.bounding_shape().numpy()
    assert len(detection_shape) == len(tracklet_shape) == 3
    # Batch size should be correct.
    assert detection_shape[0] == tracklet_shape[0] == batch_size
    # Number of features should be correct.
    assert (
        detection_shape[2]
        == tracklet_shape[2]
        == config.num_appearance_features
    )


@pytest.mark.integration
@pytest.mark.slow
def test_extract_interaction_features_smoke(faker: Faker) -> None:
    """
    Tests that `extract_interaction_features` builds a functional graph.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    image_shape = (100, 100, 3)
    batch_size = faker.random_int(min=1, max=16)
    detections = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )
    tracklets = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )

    # Create fake geometry features.
    num_geom_features = faker.random_int(min=1, max=100)
    detections_geometry = faker.ragged_tensor(
        row_lengths=detections.row_lengths(), inner_shape=(num_geom_features,)
    )
    tracklets_geometry = faker.ragged_tensor(
        row_lengths=detections.row_lengths(), inner_shape=(num_geom_features,)
    )

    config = faker.model_config(image_shape=image_shape)

    # Act.
    input_shape = (None,) + image_shape
    detection_input = tf.keras.Input(input_shape, ragged=True)
    tracklet_input = tf.keras.Input(input_shape, ragged=True)

    geom_input_shape = (None, num_geom_features)
    detection_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)
    tracklet_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)

    (
        detection_app_features,
        tracklet_app_features,
    ) = gcnn_model.extract_appearance_features(
        detections=detection_input, tracklets=tracklet_input, config=config
    )

    # Pad appearance features to dense tensors.
    to_tensor = tf.keras.layers.Lambda(lambda rt: rt.to_tensor())
    detection_app_features = to_tensor(detection_app_features)
    tracklet_app_features = to_tensor(tracklet_app_features)

    (
        tracklet_inter_features,
        detection_inter_features,
    ) = gcnn_model.extract_interaction_features(
        detections_app_features=detection_app_features,
        tracklets_app_features=tracklet_app_features,
        detections_geometry=detection_geometry_input,
        tracklets_geometry=tracklet_geometry_input,
        config=config,
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[
            detection_input,
            tracklet_input,
            detection_geometry_input,
            tracklet_geometry_input,
        ],
        outputs=[tracklet_inter_features, detection_inter_features],
    )

    # Apply the model to the inputs we generated.
    got_tracklet_features, got_detection_features = model.predict(
        (detections, tracklets, detections_geometry, tracklets_geometry)
    )

    # Assert.
    # Make sure our results are the right shape.
    detection_shape = tf.shape(got_detection_features).numpy()
    tracklet_shape = tf.shape(got_tracklet_features).numpy()
    assert len(detection_shape) == len(tracklet_shape) == 3
    # It should have the correct number of features.
    assert detection_shape[2] == tracklet_shape[2] == config.num_node_features
    # It should have the correct number of nodes.
    assert detection_shape[1] == np.max(detections.row_lengths().numpy())
    assert tracklet_shape[1] == np.max(tracklets.row_lengths().numpy())


@pytest.mark.integration
@pytest.mark.slow
def test_compute_association_smoke(faker: Faker) -> None:
    """
    Tests that `compute_association` builds a functional graph.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    app_feature_shape = (faker.random_int(max=200),)
    batch_size = faker.random_int(min=1, max=16)
    detections = faker.detected_objects(
        image_shape=app_feature_shape, batch_size=batch_size
    )
    tracklets = faker.detected_objects(
        image_shape=app_feature_shape, batch_size=batch_size
    )

    # Create fake geometry features.
    detections_geometry = faker.ragged_tensor(
        row_lengths=detections.row_lengths(), inner_shape=(4,)
    )
    tracklets_geometry = faker.ragged_tensor(
        row_lengths=tracklets.row_lengths(), inner_shape=(4,)
    )

    config = faker.model_config(image_shape=(100, 100, 3))

    # Act.
    input_shape = (None,) + app_feature_shape
    detection_input = tf.keras.Input(input_shape, ragged=True)
    tracklet_input = tf.keras.Input(input_shape, ragged=True)

    geom_input_shape = (None, 4)
    detection_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)
    tracklet_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)

    sinkhorn, assigment = gcnn_model.compute_association(
        detections_app_features=detection_input,
        tracklets_app_features=tracklet_input,
        detections_geometry=detection_geometry_input,
        tracklets_geometry=tracklet_geometry_input,
        config=config,
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[
            detection_input,
            tracklet_input,
            detection_geometry_input,
            tracklet_geometry_input,
        ],
        outputs=[sinkhorn, assigment],
    )

    # Apply the model to the inputs we generated.
    got_sinkhorn, got_assignment = model.predict(
        (detections, tracklets, detections_geometry, tracklets_geometry)
    )

    # Assert.
    # Make sure our results are the right shape.
    sinkhorn_shape = got_sinkhorn.bounding_shape()
    assignment_shape = got_assignment.bounding_shape()
    assert len(sinkhorn_shape) == len(assignment_shape) == 2

    # Make sure the association matrices are the expected size.
    row_sizes = tracklets.row_lengths().numpy()
    col_sizes = detections.row_lengths().numpy()
    expected_lengths = row_sizes * col_sizes
    np.testing.assert_array_equal(
        got_sinkhorn.row_lengths().numpy(), expected_lengths
    )
    np.testing.assert_array_equal(
        got_assignment.row_lengths().numpy(), expected_lengths
    )


@pytest.mark.integration
@pytest.mark.slow
def test_save_model_smoke(faker: Faker, tmp_path: Path) -> None:
    """
    Tests that a model can correctly be created, saved, and loaded.

    Args:
        faker: The fixture to use for generating fake data.
        tmp_path: The path to the temporary directory to use for this test.

    """
    # Arrange.
    # Create the model.
    config = faker.model_config(image_shape=(100, 100, 3))
    model = gcnn_model.build_tracking_model(config)

    save_path = tmp_path / "test_model.h5"

    # Act.
    # Save and load the model.
    model.save(save_path, save_format="h5")
    loaded_model = tf.keras.models.load_model(
        save_path, custom_objects=CUSTOM_LAYERS
    )

    # Assert.
    for weights, loaded_weights in zip(
        model.get_weights(), loaded_model.get_weights()
    ):
        np.testing.assert_array_equal(weights, loaded_weights)
