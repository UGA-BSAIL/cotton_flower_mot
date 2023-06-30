"""
Nodes for the `data_engineering` pipeline.
"""

import random
from typing import Any

import pandas as pd

from ..schemas import MotAnnotationColumns as Mot
from ..schemas import ObjectTrackingFeatures as Otf


def record_task_id(annotations: pd.DataFrame, *, task_id: int) -> pd.DataFrame:
    """
    Stores the associated CVAT task ID in the `IMAGE_SOURCE_ID` parameter
    within the annotations.

    Args:
        annotations: The annotations to add the task ID to.
        task_id: The task ID.

    Returns:
        The same annotations, with the task ID.

    """
    annotations[Otf.IMAGE_SOURCE_ID.value] = task_id
    return annotations


def merge_annotations(*args: Any) -> pd.DataFrame:
    """
    Merges annotations from multiple DataSets, setting the
    `IMAGE_SEQUENCE_ID` to differentiate between them.

    Args:
        *args: The datasets to merge from.

    Returns:
        The merged annotations.

    """
    # Set a unique sequence ID for each of them.
    sequence_id = 0
    for dataset in args:
        dataset[Otf.IMAGE_SEQUENCE_ID.value] = sequence_id
        sequence_id += 1

    # Merge them.
    return pd.concat(args, ignore_index=True)


def split_clips(
    annotations_mot: pd.DataFrame, *, max_clip_length: int
) -> pd.DataFrame:
    """
    Splits data into clips that have a specified maximum length. Will update
    the `IMAGE_SEQUENCE_ID` column to differentiate different clips.

    Args:
        annotations_mot: The annotations to split.
        max_clip_length: The maximum length in frames of the clips.

    Returns:
        The modified annotations.

    """
    old_sequence_id = -1
    new_sequence_id = -1
    old_frame_num = -1
    current_clip_length = 0

    new_sequence_ids = []
    for sequence_id, frame_num in zip(
        annotations_mot[Otf.IMAGE_SEQUENCE_ID.value],
        annotations_mot[Mot.FRAME.value],
    ):
        if sequence_id != old_sequence_id:
            # This is a new video in the underlying data.
            current_clip_length = 0
            new_sequence_id += 1
            # Our frame numbers will start over here.
            old_frame_num = 0
        old_sequence_id = sequence_id

        if frame_num != old_frame_num:
            # Only count frames as opposed to annotations.
            # The subtraction is because some frames might have no annotations
            # and are thus skipped.
            current_clip_length += frame_num - old_frame_num
            old_frame_num = frame_num

        if current_clip_length >= max_clip_length:
            # We should split a new clip here.
            current_clip_length = 0
            new_sequence_id += 1

        new_sequence_ids.append(new_sequence_id)

    # Set the new sequence IDs.
    annotations_mot[Otf.IMAGE_SEQUENCE_ID.value] = new_sequence_ids

    return annotations_mot


def shuffle_clips(annotations_mot: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffles the order of clips.

    Args:
        annotations_mot: The annotations to shuffle. They should already have
            been split into clips.

    Returns:
        The same annotations, with the clips shuffled.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations_mot.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    clip_annotations = []
    for sequence_id in annotations_mot.index.unique():
        clip_annotations.append(
            annotations_mot.iloc[annotations_mot.index == sequence_id]
        )

    # Shuffle the clips.
    random.shuffle(clip_annotations)
    # Re-compose them back into a single dataframe.
    return pd.concat(clip_annotations, ignore_index=True)


def _transform_bounding_boxes(mot_annotations: pd.DataFrame) -> None:
    """
    Transforms bounding boxes from the x, y, width, height format that MOT 1.1
    uses to the x_min, x_max, y_min, y_max format that we use in TensorFlow.

    Args:
        mot_annotations: The annotations, in MOT 1.1 format. They will be
            modified in-place. It will use the TensorFlow column names.

    """
    x_min = mot_annotations[Mot.BBOX_X_MIN_PX.value]
    y_min = mot_annotations[Mot.BBOX_Y_MIN_PX.value]
    x_max = x_min + mot_annotations[Mot.BBOX_WIDTH_PX.value]
    y_max = y_min + mot_annotations[Mot.BBOX_HEIGHT_PX.value]

    # Use the right column names.
    mot_annotations.rename(
        {
            Mot.BBOX_X_MIN_PX.value: Otf.OBJECT_BBOX_X_MIN.value,
            Mot.BBOX_Y_MIN_PX.value: Otf.OBJECT_BBOX_Y_MIN.value,
        },
        axis=1,
        inplace=True,
    )
    # Add the additional columns.
    mot_annotations[Otf.OBJECT_BBOX_X_MAX.value] = x_max
    mot_annotations[Otf.OBJECT_BBOX_Y_MAX.value] = y_max
    # Remove the extraneous columns.
    mot_annotations.drop(
        [Mot.BBOX_WIDTH_PX.value, Mot.BBOX_HEIGHT_PX.value],
        axis=1,
        inplace=True,
    )


def mot_to_object_detection_format(
    mot_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts annotations from the MOT 1.1 format to a format format that's
    more compatible with the TF Object Detection API.

    Args:
        mot_annotations: The raw annotations, in MOT 1.1 format.

    Returns:
        The transformed annotations.

    """
    # Update the bounding box format.
    _transform_bounding_boxes(mot_annotations)

    # Rename the remaining columns.
    mot_annotations.rename(
        {
            Mot.FRAME.value: Otf.IMAGE_FRAME_NUM.value,
            Mot.ID.value: Otf.OBJECT_ID.value,
        },
        axis=1,
        inplace=True,
    )
    # Drop columns that we don't care about.
    mot_annotations.drop(
        [
            Mot.CONFIDENCE.value,
            Mot.CLASS_ID.value,
            Mot.VISIBILITY.value,
        ],
        axis=1,
        inplace=True,
    )

    return mot_annotations
