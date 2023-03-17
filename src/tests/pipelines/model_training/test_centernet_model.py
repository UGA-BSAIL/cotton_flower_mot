"""
Tests for the `centernet_model` module.
"""


from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import centernet_model
from src.cotton_flower_mot.pipelines.model_training.layers import CUSTOM_LAYERS


# TODO (danielp) Re-enable this test once offsets work again.
@pytest.mark.skip
@pytest.mark.parametrize(
    ("confidence_mask", "sizes", "offsets", "expected_boxes"),
    [
        (
            np.array([[0, 0.75], [0.9, 0]]),
            np.array(
                [[[0.25, 0.20], [0.65, 0.72]], [[0.09, 0.43], [0.31, 0.08]]]
            ),
            np.array(
                [
                    [[0.001, 0.003], [0.002, 0.0012]],
                    [[0.01, 0.002], [0.005, 0.0043]],
                ]
            ),
            np.array(
                [
                    [0.502, 0.0012, 0.65, 0.72, 0.75],
                    [0.01, 0.502, 0.09, 0.43, 0.9],
                ]
            ),
        )
    ],
    ids=["standard"],
)
def test_compute_sparse_predictions(
    confidence_mask: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    expected_boxes: np.ndarray,
) -> None:
    """
    Tests that `compute_sparse_predictions` works.

    Args:
        confidence_mask: The mask for the center pixels to test with.
        sizes: The sizes for each pixel in the heatmap.
        offsets: The offsets for each pixel in the heatmap.
        expected_boxes: The expected resulting bounding boxes.

    """
    # Arrange.
    # Duplicate along the batch dimension so we can test batching.
    confidence_mask = tf.constant(np.stack([confidence_mask] * 2, 0))
    confidence_mask = tf.expand_dims(confidence_mask, -1)
    sizes = tf.constant(np.stack([sizes.astype(np.float32)] * 2, 0))
    offsets = tf.constant(np.stack([offsets.astype(np.float32)] * 2, 0))
    expected_boxes = tf.constant(np.stack([expected_boxes] * 2, 0))

    # Act.
    got_boxes = centernet_model.compute_sparse_predictions(
        confidence_masks=confidence_mask, sizes=sizes, offsets=offsets
    )

    # Assert.
    got_boxes = got_boxes.to_tensor().numpy()
    np.testing.assert_array_almost_equal(got_boxes, expected_boxes)


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
    config = faker.model_config(detection_input_shape=(540, 960, 3))
    _, model = centernet_model.build_detection_model(config)

    save_path = tmp_path / "test_model.h5"

    # Act.
    # Save and load the model.
    model.save(save_path, save_format="h5", include_optimizer=False)
    loaded_model = tf.keras.models.load_model(
        save_path, custom_objects=CUSTOM_LAYERS, compile=False
    )

    # Assert.
    for weights, loaded_weights in zip(
        model.get_weights(), loaded_model.get_weights()
    ):
        np.testing.assert_array_equal(weights, loaded_weights)
