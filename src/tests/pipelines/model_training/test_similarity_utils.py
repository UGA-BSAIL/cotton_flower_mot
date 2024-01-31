from typing import List

import pytest
import tensorflow as tf
import yaml
from pytest_snapshot.plugin import Snapshot

from src.cotton_flower_mot import similarity_utils


@pytest.mark.parametrize(
    ["box1", "box2", "iou"],
    [
        ([5.0, 10.0, 4.0, 5.0], [20.0, 20.0, 6.0, 6.0], 0.0),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 6.0], 1.0),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 3.0], 0.5),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 12.0, 5.0, 6.0], 0.5),
        ([10.0, 10.0, 4.0, 6.0], [12.0, 13.0, 4.0, 6.0], 1.0 / 7.0),
        ([10.0, 10.0, 4.0, 6.0], [12.0, 7.0, 4.0, 6.0], 1.0 / 7.0),
        ([10.0, 10.0, 4.0, 6.0], [8.0, 7.0, 4.0, 6.0], 1.0 / 7.0),
        ([10.0, 10.0, 4.0, 6.0], [8.0, 13.0, 4.0, 6.0], 1.0 / 7.0),
        ([0.0, 0.0, 4.0, 6.0], [-2.0, -3.0, 4.0, 6.0], 1.0 / 7.0),
    ],
    ids=[
        "no_overlap",
        "perfect_overlap",
        "same_center",
        "one_dimension",
        "above_right",
        "below_right",
        "below_left",
        "above_left",
        "negative",
    ],
)
def test_compute_ious(
    box1: List[float], box2: List[float], iou: float
) -> None:
    """
    Tests that `compute_ious` works.

    Args:
        box1: The first bounding box.
        box2: The second bounding box.
        iou: The expected IOU.

    """
    # Arrange.
    # Convert bounding boxes to tensors.
    box1 = tf.constant([box1])
    box2 = tf.constant([box2])

    # Act.
    got_iou = similarity_utils.compute_ious(box1, box2).numpy()

    # Assert.
    assert got_iou.squeeze() == pytest.approx(iou)


@pytest.mark.parametrize(
    ["box1", "box2"],
    [
        ([5.0, 10.0, 4.0, 5.0], [20.0, 20.0, 6.0, 6.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 6.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 3.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 12.0, 5.0, 6.0]),
        ([10.0, 10.0, 4.0, 6.0], [12.0, 13.0, 4.0, 6.0]),
        ([10.0, 10.0, 4.0, 6.0], [12.0, 7.0, 4.0, 6.0]),
        ([10.0, 10.0, 4.0, 6.0], [8.0, 7.0, 4.0, 6.0]),
        ([10.0, 10.0, 4.0, 6.0], [8.0, 13.0, 4.0, 6.0]),
        ([0.0, 0.0, 4.0, 6.0], [-2.0, -3.0, 4.0, 6.0]),
    ],
    ids=[
        "no_overlap",
        "perfect_overlap",
        "same_center",
        "one_dimension",
        "above_right",
        "below_right",
        "below_left",
        "above_left",
        "negative",
    ],
)
def test_distance_penalty(
    box1: List[float], box2: List[float], snapshot: Snapshot
) -> None:
    """
    Tests that `distance_penalty` works.

    Args:
        box1: The first bounding box.
        box2: The second bounding box.
        snapshot: The fixture to use for snapshot testing.

    """
    # Arrange.
    # Convert bounding boxes to tensors.
    box1 = tf.constant([box1])
    box2 = tf.constant([box2])

    # Act.
    got_penalty = similarity_utils.distance_penalty(box1, box2).numpy()

    # Assert.
    snapshot.assert_match(
        yaml.dump(got_penalty.tolist()), "distance_penalty.yml"
    )


@pytest.mark.parametrize(
    ["box1", "box2"],
    [
        ([5.0, 10.0, 4.0, 5.0], [20.0, 20.0, 4.0, 5.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 6.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 5.0, 3.0]),
        ([5.0, 10.0, 4.0, 6.0], [20.0, 20.0, 2.0, 3.0]),
        ([10.0, 10.0, 5.0, 6.0], [10.0, 10.0, 6.0, 5.0]),
        ([0.0, 0.0, 4.0, 6.0], [-2.0, -3.0, 4.0, 6.0]),
    ],
    ids=[
        "same_size",
        "perfect_overlap",
        "same_center",
        "same_aspect_ratio",
        "flipped_aspect_ratio",
        "negative",
    ],
)
def test_aspect_ratio_penalty(
    box1: List[float], box2: List[float], snapshot: Snapshot
) -> None:
    """
    Tests that `aspect_ratio_penalty` works.

    Args:
        box1: The first bounding box.
        box2: The second bounding box.
        snapshot: The fixture to use for snapshot testing.

    """
    # Arrange.
    # Convert bounding boxes to tensors.
    box1 = tf.constant([box1])
    box2 = tf.constant([box2])

    # Act.
    got_penalty = similarity_utils.aspect_ratio_penalty(box1, box2).numpy()

    # Assert.
    snapshot.assert_match(
        yaml.dump(got_penalty.tolist()), "aspect_ratio_penalty.yml"
    )


@pytest.mark.parametrize(
    ["feature1", "feature2", "similarity"],
    [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1.0),
        ([1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], -1.0),
    ],
    ids=["perfect_similarity", "perfect_dissimilarity"],
)
def test_cosine_similarity(
    feature1: List[float], feature2: List[float], similarity: float
) -> None:
    """
    Tests that `cosine_similarity` works.

    Args:
        feature1: The first feature vector.
        feature2: The second feature vector.
        similarity: The expected similarity between the two.

    """
    # Arrange.
    # Convert features to tensors.
    feature1 = tf.constant([feature1])
    feature2 = tf.constant([feature2])

    # Act.
    got_similarity = similarity_utils.cosine_similarity(
        feature1, feature2
    ).numpy()

    # Assert.
    assert got_similarity.squeeze() == pytest.approx(similarity)
