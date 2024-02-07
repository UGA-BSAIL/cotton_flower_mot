import random
from logging import DEBUG
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from swagger_client.rest import ApiException
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from pycvat import Task

from ..config import ModelConfig
from ..heat_maps import make_object_heat_map
from ...schemas import ObjectTrackingFeatures as Otf
from ..tfrecords_utils import bytes_feature, float_feature, int_feature

_FEATURES_TO_FACTORIES = {
    Otf.IMAGE_HEIGHT: int_feature,
    Otf.IMAGE_WIDTH: int_feature,
    Otf.IMAGE_FILENAME: bytes_feature,
    Otf.IMAGE_SOURCE_ID: int_feature,
    Otf.IMAGE_ENCODED: bytes_feature,
    Otf.IMAGE_FORMAT: bytes_feature,
    Otf.OBJECT_BBOX_X_MIN: float_feature,
    Otf.OBJECT_BBOX_X_MAX: float_feature,
    Otf.OBJECT_BBOX_Y_MIN: float_feature,
    Otf.OBJECT_BBOX_Y_MAX: float_feature,
    Otf.OBJECT_CLASS_TEXT: bytes_feature,
    Otf.OBJECT_CLASS_LABEL: int_feature,
    Otf.OBJECT_ID: int_feature,
    Otf.IMAGE_SEQUENCE_ID: int_feature,
    Otf.IMAGE_FRAME_NUM: int_feature,
    Otf.HEATMAP_ENCODED: bytes_feature,
}


def split_random(
    annotations: pd.DataFrame, *, train_fraction: float, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits annotation data into training and testing components.

    Args:
        annotations: The annotations to split. Should already be organized
            into clips.
        train_fraction: The fraction of the data to use for training.
        test_fraction: The fraction of the data to use for testing.

    Returns:
        The individual train, test, and validation datasets.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    train_clips = []
    test_clips = []
    validation_clips = []
    for sequence_id in annotations.index.unique():
        clip = annotations.iloc[annotations.index == sequence_id]

        sample = random.random()
        if sample > train_fraction + test_fraction:
            validation_clips.append(clip)
        elif sample > train_fraction:
            test_clips.append(clip)
        else:
            train_clips.append(clip)

    # Concatenate the clips back into single dataframes.
    def concat(clips: List[pd.DataFrame]) -> pd.DataFrame:
        if len(clips) > 0:
            return pd.concat(clips, ignore_index=True)
        else:
            return pd.DataFrame([], columns=[e.value for e in Otf])

    return concat(train_clips), concat(test_clips), concat(validation_clips)


def split_specific(
    annotations: pd.DataFrame,
    *,
    test_clips: Iterable[int],
    valid_clips: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split out specific training, testing, and validation clips.

    Args:
        annotations: The complete annotations.
        test_clips: The specific sequence IDs to use for testing.
        valid_clips: The specific sequence IDs to use for validation.

    Returns:
        The individual train, test, and validation datasets.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    testing_split = annotations.iloc[annotations.index.isin(test_clips)]
    validation_split = annotations.iloc[annotations.index.isin(valid_clips)]
    train_split = annotations.iloc[
        ~annotations.index.isin(valid_clips)
        & ~annotations.index.isin(test_clips)
    ]

    return train_split, testing_split, validation_split


def _get_missing_columns(
    frame_annotations: pd.DataFrame,
) -> Dict[str, tf.train.Feature]:
    """
    Gets a dictionary containing default values for any missing features.

    Args:
        frame_annotations: The frame annotations.

    Returns:
        A dict containing the default values.

    """
    default_values = {
        int_feature: [-1],
        bytes_feature: b"",
        float_feature: [0.0],
    }

    missing_features = {}
    for column, feature_type in _FEATURES_TO_FACTORIES.items():
        if column.value not in frame_annotations:
            missing_features[column.value] = feature_type(
                default_values[feature_type]
            )

    return missing_features


def _make_example(
    *,
    image: np.ndarray,
    heat_map: np.ndarray,
    frame_annotations: pd.DataFrame,
    frame_num: int,
    sequence_id: int,
) -> tf.train.Example:
    """
    Creates a TF `Example` for a single frame.

    Args:
        image: The compressed image data for the frame.
        heat_map: Detection heatmap associated with image.
        frame_annotations: The annotations for the frame.
        frame_num: The frame number.
        sequence_id: The sequence ID.

    Returns:
        The example that it created.

    """
    # Shuffle the order of the rows to add some variation to Sinkhorn matrices.
    frame_annotations = frame_annotations.sample(frac=1.0)

    # Remove the frame number and sequence ID columns since that info is
    # provided manually.
    frame_annotations = frame_annotations.drop(
        columns=[Otf.IMAGE_FRAME_NUM.value, Otf.IMAGE_SEQUENCE_ID.value]
    )

    # Create the feature dictionary.
    features = {}
    for column_name, column_data in frame_annotations.items():
        features[column_name] = _FEATURES_TO_FACTORIES[Otf(column_name)](
            column_data
        )

    # Add default values for missing columns.
    features.update(_get_missing_columns(frame_annotations))

    # Add the image data.
    features[Otf.IMAGE_ENCODED.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_ENCODED
    ](image)
    features[Otf.HEATMAP_ENCODED.value] = _FEATURES_TO_FACTORIES[
        Otf.HEATMAP_ENCODED
    ](heat_map)
    # Add the frame number and sequence ID.
    features[Otf.IMAGE_FRAME_NUM.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_FRAME_NUM
    ]((frame_num,))
    features[Otf.IMAGE_SEQUENCE_ID.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_SEQUENCE_ID
    ]((sequence_id,))

    return tf.train.Example(features=tf.train.Features(feature=features))


def _generate_heat_map(
    frame_annotations: pd.DataFrame, *, config: ModelConfig
) -> np.ndarray:
    """
    Generates a detection heatmap given input annotations.

    Args:
        frame_annotations: The annotations for this frame.
        config: The model configuration.

    Returns:
        The compressed heatmap that it produced.

    """
    # Calculate the heatmap size.
    heatmap_size = np.array(config.heatmap_size)

    # Extract the detection centers.
    image_height, image_width, _ = config.frame_input_shape
    frame_size = np.array([image_width, image_height])
    high_points = frame_annotations[
        [Otf.OBJECT_BBOX_X_MAX.value, Otf.OBJECT_BBOX_Y_MAX.value]
    ].values
    low_points = frame_annotations[
        [Otf.OBJECT_BBOX_X_MIN.value, Otf.OBJECT_BBOX_Y_MIN.value]
    ].values
    sizes = high_points - low_points
    center_points = low_points + sizes / 2
    normalized_center_points = center_points / frame_size
    normalized_sizes = sizes / frame_size
    boxes = tf.concat(
        (
            tf.convert_to_tensor(normalized_center_points, dtype=tf.float32),
            tf.convert_to_tensor(normalized_sizes, dtype=tf.float32),
        ),
        axis=1,
    )

    # Create the heatmap.
    heat_map = make_object_heat_map(
        boxes,
        map_size=tf.convert_to_tensor(heatmap_size, dtype=tf.int32),
        normalized=False,
    )

    # Encode the heatmap as integers so that we can store it as a PNG.
    heat_map *= np.iinfo(np.uint16).max
    heat_map = tf.cast(heat_map, tf.uint16)
    # Compress the heatmap.
    return tf.io.encode_png(heat_map).numpy()


@retry(
    retry=retry_if_exception_type(ApiException),
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(20),
    after=after_log(logger, DEBUG),
)
def _get_frame_image(video_frames: Task, *, frame_num: int) -> np.ndarray:
    """
    Gets a single frame for CVAT.

    Args:
        video_frames: The CVAT task to pull frames from.
        frame_num: The frame number to get.

    Returns:
        The frame, as a JPEG.

    """
    frame_image = video_frames.get_image(frame_num)
    # CVAT data is generally compressed as PNGs, which are not ideal for
    # this application.
    encoded, frame_image = cv2.imencode(".jpg", frame_image)
    assert encoded, "Failed to encode image."

    return frame_image


def _generate_clip_examples(
    video_frames: Task, annotations: pd.DataFrame, *, config: ModelConfig
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from annotations and corresponding video frames.

    Args:
        video_frames: The CVAT `Task` to source frames from.
        annotations: The loaded annotations, transformed to the TF format.
        config: The model configuration.

    Yields:
        Corresponding TFRecord examples for each frame. The examples are
        correspond to the TF Object Detection API format, except with the
        addition of an "image/object/id" label to group detections that
        are part of the same track.

    """
    frame_nums = annotations[Otf.IMAGE_FRAME_NUM.value]
    first_frame = frame_nums.min()
    last_frame = frame_nums.max()

    # Get the sequence ID for this clip.
    sequence_id = annotations[Otf.IMAGE_SEQUENCE_ID.value].unique()[0]
    logger.debug("Sequence ID: {}", sequence_id)

    for frame_num in range(first_frame, last_frame + 1):
        logger.debug("Generating example for frame {}.", frame_num)

        # Get all the annotations for that frame.
        frame_annotations = annotations[
            annotations[Otf.IMAGE_FRAME_NUM.value] == frame_num
        ]
        logger.debug(
            "Have {} annotations for frame {}.",
            len(frame_annotations),
            frame_num,
        )

        # Get the actual frame image.
        frame_image = _get_frame_image(video_frames, frame_num=frame_num)

        # Get the heatmap.
        heat_map = _generate_heat_map(frame_annotations, config=config)

        yield _make_example(
            image=frame_image,
            frame_annotations=frame_annotations,
            frame_num=frame_num,
            heat_map=heat_map,
            sequence_id=sequence_id,
        )


def generate_examples(
    video_frames: Iterable[Task],
    annotations: pd.DataFrame,
    *,
    config: ModelConfig,
) -> Iterable[Iterable[tf.train.Example]]:
    """
    Generates TFRecord examples from annotations and corresponding video
    frames for all clips.

    Args:
        video_frames: The CVAT `Task`s to source frames from.
        annotations: The loaded annotations, transformed to the TF format.
        config: The model configuration.

    Yields:
        Iterables of TFRecord examples for each clip.

    """
    # Associate tasks with their corresponding ID.
    id_to_task = {task.id: task for task in video_frames}

    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    for sequence_id in annotations.index.unique():
        logger.info("Generating TFRecords for clip {}.", sequence_id)

        sequence_annotations = annotations.iloc[
            annotations.index == sequence_id
        ]
        task_id = int(
            sequence_annotations[Otf.IMAGE_SOURCE_ID.value].unique().squeeze()
        )
        logger.debug("Using frames from task {}.", task_id)

        yield _generate_clip_examples(
            id_to_task[task_id],
            annotations.iloc[annotations.index == sequence_id],
            config=config,
        )
