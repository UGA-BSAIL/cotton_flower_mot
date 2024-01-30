"""
Nodes for the annotation conversion pipeline. This pipeline converts annotations
from other styles to MOT 1.1, which is the standard for this project.
"""


from typing import Iterable, Tuple

import pandas as pd

from src.cotton_flower_mot.schemas import MotAnnotationColumns as Mot


def merge_frame_annotations(
    annotations: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merges individual sets of annotations from sequential frames into a single
    `DataFrame`. It will add a "frame" column to distinguish which frame the
    data came from.

    Args:
        annotations: The annotations for each frame, in order, as individual
            DFs.

    Returns:
        The combined annotation data.

    """
    # Add the frame number to individual data frames.
    def _with_frame() -> Iterable[pd.DataFrame]:
        for frame_num, frame_annotations in enumerate(annotations):
            if frame_annotations is None:
                # There are no annotations provided for this frame.
                continue

            frame_annotations[Mot.FRAME.value] = frame_num
            yield frame_annotations

    # Combine them all.
    return pd.concat(_with_frame(), ignore_index=True)


def _bboxes_to_mot_format(
    annotations: pd.DataFrame, *, image_size: Tuple[int, int]
) -> None:
    """
    Converts the bounding boxes in a set of annotations to MOT 1.1 format.

    Args:
        annotations: The combined annotations, in Chenjiao's format. Will be
            modified in-place.
        image_size: Since bounding boxes are in terms of frame fractions, we
            need to specify the size of the images in terms of (width, height).

    """
    # Specify the bounding box in the correct way.
    image_width, image_height = image_size
    bbox_width_px = annotations["width"] * image_width
    bbox_height_px = annotations["height"] * image_height
    bbox_x_min_px = annotations["center_x"] * image_width - bbox_width_px / 2
    bbox_y_min_px = annotations["center_y"] * image_height - bbox_height_px / 2

    annotations[Mot.BBOX_X_MIN_PX.value] = bbox_x_min_px
    annotations[Mot.BBOX_Y_MIN_PX.value] = bbox_y_min_px
    annotations[Mot.BBOX_WIDTH_PX.value] = bbox_width_px
    annotations[Mot.BBOX_HEIGHT_PX.value] = bbox_height_px

    # Drop the originals.
    annotations.drop(
        ["width", "height", "center_x", "center_y"], axis=1, inplace=True
    )


def convert_to_mot_format(
    annotations: pd.DataFrame, *, image_size: Tuple[int, int]
) -> pd.DataFrame:
    """
    Converts annotations to MOT 1.1 format.

    Args:
        annotations: The combined annotations, in Chenjiao's format. Will be
            modified in-place.
        image_size: Since bounding boxes are in terms of frame fractions, we
            need to specify the size of the images in terms of (width, height).

    Returns:
        The same annotations in MOT 1.1 format.

    """
    # Convert bounding boxes.
    _bboxes_to_mot_format(annotations, image_size=image_size)

    # The other columns can be copied straight over or dropped.
    annotations.rename({"id": Mot.ID.value}, axis=1, inplace=True)
    annotations.drop(["class"], axis=1, inplace=True)

    # Fill extraneous columns with default values.
    annotations[Mot.CONFIDENCE.value] = 1
    annotations[Mot.CLASS_ID.value] = 1
    annotations[Mot.VISIBILITY.value] = 1.0

    return annotations
