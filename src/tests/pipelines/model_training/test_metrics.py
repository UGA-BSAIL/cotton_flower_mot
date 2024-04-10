"""
Tests for the `metrics` module.
"""

import numpy as np
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import metrics


def test_average_precision_smoke(faker: Faker) -> None:
    """
    Tests that the `AveragePrecision` metric doesn't crash when we give it
    valid data.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    # Create fake ground truth and predictions.
    true_row_lengths = faker.random_choices(list(range(10)), length=10)
    pred_row_lengths = faker.random_choices(list(range(10)), length=10)
    # Force a test for the case where the row length is zero.
    pred_row_lengths.append(0)
    true_row_lengths.append(1)
    y_true = faker.ragged_tensor(
        row_lengths=true_row_lengths,
        inner_shape=(6,),
        min_value=0.0,
        max_value=1.0,
    )
    y_pred = faker.ragged_tensor(
        row_lengths=pred_row_lengths,
        inner_shape=(5,),
        min_value=0.0,
        max_value=1.0,
    )

    # Act.
    # Deliberately reduce the IOU threshold so it's more likely to find matches.
    metric = metrics.AveragePrecision(iou_threshold=0.1)
    metric.update_state(y_true, y_pred)

    # Assert.
    # We should be able to get a result that's not stupid.
    result = metric.result().numpy()
    assert not np.any(np.isnan(result))
    assert 0.0 <= result <= 1.0
