"""
Utilities for handling the MOT challenge data format.
"""
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd

from .online_tracker import Track
from .schemas import MotAnnotationColumns


def _filter_extrapolated_track_end(track: Track) -> pd.DataFrame:
    """
    Filters the part of the track at the end that has been extrapolated by
    the motion model.

    Args:
        track: The track to filter.

    Returns:
        The filtered track.

    """
    if track.last_detection_frame is None:
        # No detections.
        return track.all_detections()

    return track.all_detections().loc[: (track.last_detection_frame + 1)]


def track_to_mot_challenge(
    track: Track,
    resolution: Tuple[int, int],
    cvat: bool = False,
) -> pd.DataFrame:
    """
    Converts a track to the MOT Challenge format.

    Args:
        track: The track to convert.
        resolution: The width and height of the clip.
        cvat: Whether to use the CVAT flavor of this annotation format.

    Returns:
        The track in the MOT Challenge format.

    """
    detections = _filter_extrapolated_track_end(track)
    has_detection = np.array(
        [track.has_real_detection_for_frame(f) for f in detections.index]
    )

    # It wants bounding boxes in a different format, so fix that.
    frame_width, frame_height = resolution
    bbox_min_x = detections["center_x"] - detections["width"] / 2
    bbox_min_y = detections["center_y"] - detections["height"] / 2
    # Convert to pixels.
    bbox_min_x *= frame_width
    bbox_min_y *= frame_height

    box_width = detections["width"] * frame_width
    box_height = detections["height"] * frame_height

    if not cvat:
        # For confidence, we'll just set it to one for actual detections and
        # zero for extrapolated bounding boxes.
        confidence = has_detection.astype(float)
    else:
        confidence = 1.0

    annotation_data = OrderedDict(
        [
            (MotAnnotationColumns.FRAME.value, detections.index),
            (MotAnnotationColumns.ID.value, track.id),
            (MotAnnotationColumns.BBOX_X_MIN_PX.value, bbox_min_x),
            (MotAnnotationColumns.BBOX_Y_MIN_PX.value, bbox_min_y),
            (MotAnnotationColumns.BBOX_WIDTH_PX.value, box_width),
            (MotAnnotationColumns.BBOX_HEIGHT_PX.value, box_height),
        ]
    )
    if cvat:
        annotation_data[MotAnnotationColumns.CLASS_ID.value] = 1
        annotation_data[MotAnnotationColumns.VISIBILITY.value] = 1
        # Confidence goes at the end.
        annotation_data[MotAnnotationColumns.CONFIDENCE.value] = confidence
    else:
        annotation_data[MotAnnotationColumns.CONFIDENCE.value] = confidence
        # These are for 3D tracking, which we're not doing.
        annotation_data["x"] = -1
        annotation_data["y"] = -1
        annotation_data["z"] = -1

    return pd.DataFrame(data=annotation_data)
