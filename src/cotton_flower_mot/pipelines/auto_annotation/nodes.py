"""
Annotates data automatically using a trained model.
"""


import enum
from logging import DEBUG
from typing import Tuple

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

from src.cotton_flower_mot.schemas import MotAnnotationColumns


@enum.unique
class DetectionColumns(enum.Enum):
    """
    Represents the columns used when storing raw detections.
    """

    CENTER_X = "center_x"
    """
    X-coordinate of the center point.
    """
    CENTER_Y = "center_y"
    """
    Y-coordinate of the center point.
    """
    WIDTH = "width"
    """
    Bounding box width.
    """
    HEIGHT = "height"
    """
    Bounding box height.
    """
    CONFIDENCE = "confidence"
    """
    Detection confidence.
    """
    FRAME = "frame"
    """
    The frame number corresponding to this detection.
    """


def _non_max_suppression(
    boxes: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Performs non-maximum suppression on a set of bounding boxes.

    Args:
        boxes: The bounding boxes to perform NMS on, in the form
            `[center_x, center_y, width, height, confidence]`.

    Returns:
        The filtered bounding boxes.

    """
    # Convert to the form used by TensorFlow.
    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    confidence = boxes[:, 4]

    min_x = center_x - width / 2.0
    min_y = center_y - height / 2.0
    max_x = center_x + width / 2.0
    max_y = center_y + height / 2.0
    tf_boxes = np.stack((min_x, min_y, max_x, max_y), axis=1)

    # Run NMS.
    filtered_indices = tf.image.non_max_suppression(
        tf_boxes, confidence, len(confidence), iou_threshold=iou_threshold
    ).numpy()

    # Get the original boxes.
    return boxes[filtered_indices]


def _filter_all_overlapping(
    detections: pd.DataFrame, iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Filters all overlapping boxes from a dataframe of detections, based on an
    IOU threshold.

    Args:
        detections: The dataframe containing the raw detections.
        iou_threshold: The IOU threshold.

    Returns:
        The filtered detections.

    """
    frames = detections[DetectionColumns.FRAME.value].unique()

    all_filtered_detections = []
    detection_columns = [
        DetectionColumns.CENTER_X.value,
        DetectionColumns.CENTER_Y.value,
        DetectionColumns.WIDTH.value,
        DetectionColumns.HEIGHT.value,
        DetectionColumns.CONFIDENCE.value,
    ]
    for frame_num in frames:
        # Find all detections for this frame.
        frame_detections = detections[
            detections[DetectionColumns.FRAME.value] == frame_num
        ]

        detections_array = frame_detections[detection_columns].to_numpy()
        filtered_detections = _non_max_suppression(
            detections_array, iou_threshold=iou_threshold
        )

        # Add the frame column back.
        filtered_detections = pd.DataFrame(
            filtered_detections, columns=detection_columns
        )
        filtered_detections[DetectionColumns.FRAME.value] = frame_num
        all_filtered_detections.append(filtered_detections)

    all_filtered_detections = pd.concat(
        all_filtered_detections, ignore_index=True
    )
    logger.debug(
        "Filtered {} redundant boxes.",
        len(detections) - len(all_filtered_detections),
    )
    return all_filtered_detections


def _to_mot_format(
    detections: pd.DataFrame, *, frame_size: Tuple[int, int]
) -> pd.DataFrame:
    """
    Converts detections from the format returned by the model to the MOT 1.1
    annotation format.

    Args:
        detections: The model detections.
        frame_size: The size of the input frame. This is what allows us to
            convert from normalized to pixel coordinates. Should be in the
            form `(width, height)`.

    Returns:
        The same detections, in MOT format.

    """
    center_x = detections[DetectionColumns.CENTER_X.value]
    center_y = detections[DetectionColumns.CENTER_Y.value]
    width = detections[DetectionColumns.WIDTH.value]
    height = detections[DetectionColumns.HEIGHT.value]
    frames = detections[DetectionColumns.FRAME.value]

    bbox_x_min = center_x - width / 2.0
    bbox_y_min = center_y - height / 2.0

    # Convert to pixel coordinates.
    mot_data = np.stack((bbox_x_min, bbox_y_min, width, height), axis=1)
    # Convert to pixel coordinates.
    frame_width, frame_height = frame_size
    frame_width_height = np.array([frame_width, frame_height] * 2)
    mot_data *= frame_width_height

    mot_data = pd.DataFrame(
        mot_data,
        columns=[
            MotAnnotationColumns.BBOX_X_MIN_PX.value,
            MotAnnotationColumns.BBOX_Y_MIN_PX.value,
            MotAnnotationColumns.BBOX_WIDTH_PX.value,
            MotAnnotationColumns.BBOX_HEIGHT_PX.value,
        ],
    )
    # Add frame data back.
    mot_data[MotAnnotationColumns.FRAME.value] = frames.reset_index(drop=True)
    return mot_data


def _get_frame_detections(
    frame: np.ndarray,
    *,
    model: tf.keras.Model,
    num_to_keep: int = 100,
) -> pd.DataFrame:
    """
    Gets the detections for a particular frame.

    Args:
        frame: The frame to extract detections from.
        model: The model to use for making predictions.
        num_to_keep: Maximum number of detections to save. All other detections
            with lower confidence will be discarded.

    Returns:
        The extracted detections, in a `DataFrame`.

    """
    # Make sure input is the correct shape.
    frame_batch = np.expand_dims(frame, 0)
    input_size = model.input_shape[1:3]
    logger.debug("Got model input shape: {}", input_size)
    frame_batch = tf.image.resize(frame_batch, input_size)

    # Get the outputs.
    model_outputs = model(frame_batch, training=False)
    bbox_predictions = model_outputs[2][0].numpy()

    # Take only the top 100 predictions.
    confidence = bbox_predictions[..., 4]
    confidence_order = np.argsort(confidence)[::-1]
    bbox_predictions = bbox_predictions[confidence_order[:num_to_keep]]
    logger.debug(
        "Max confidence is {}.",
        confidence.max(),
    )

    return pd.DataFrame(
        data=bbox_predictions,
        columns=[
            DetectionColumns.CENTER_X.value,
            DetectionColumns.CENTER_Y.value,
            DetectionColumns.WIDTH.value,
            DetectionColumns.HEIGHT.value,
            DetectionColumns.CONFIDENCE.value,
        ],
    )


def convert_to_mot(
    detections: pd.DataFrame,
    *,
    video: Task,
    min_confidence: float = 0.5,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Takes the raw detections, post-processes them, and converts them to the
    MOT 1.1 format.

    Args:
        detections: The raw detections.
        min_confidence: Any predictions that are below this confidence
            threshold will be discarded.
        iou_threshold: The IOU threshold to use for NMS.
        video: The corresponding video that these data are from.

    Returns:
        The detections, in MOT 1.1 format.

    """
    # Discard any predictions with low confidence.
    confidence = detections[DetectionColumns.CONFIDENCE.value]
    detections = detections[confidence >= min_confidence]
    logger.debug("Got {} object detections.", len(detections))

    # Filter overlapping detections.
    detections = _filter_all_overlapping(
        detections, iou_threshold=iou_threshold
    )

    # Get the size of the video frames.
    frame_size = video.get_image_size(0)
    logger.debug("Got frame size for video: {}", frame_size)

    mot_detections = _to_mot_format(detections, frame_size=frame_size)

    # Since we do only detections, we treat each one as a separate track.
    detection_ids = np.arange(1, len(mot_detections) + 1)
    mot_detections[MotAnnotationColumns.ID.value] = detection_ids

    # Fill in extraneous columns with default values.
    mot_detections[MotAnnotationColumns.CLASS_ID.value] = 1
    mot_detections[MotAnnotationColumns.CONFIDENCE.value] = 1
    mot_detections[MotAnnotationColumns.VISIBILITY.value] = 1.0

    return mot_detections


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
        The frame data.

    """
    return video_frames.get_image(frame_num)


def annotate_video(video: Task, *, model: tf.keras.Model) -> pd.DataFrame:
    """
    Annotates a complete video using a pretrained model.

    Args:
        video: The video to annotate.
        model: The model to use for annotation.

    Returns:
        The complete annotations, in MOT 1.1 format.

    """
    num_frames = video.num_frames
    logger.debug("Video has {} frames.", num_frames)

    all_detections = []
    for frame_num in range(num_frames):
        # Get the frame image.
        frame = _get_frame_image(video, frame_num=frame_num)

        # Run inference.
        detections = _get_frame_detections(frame, model=model)
        # Frames are 1-indexed.
        detections[MotAnnotationColumns.FRAME.value] = frame_num + 1

        all_detections.append(detections)

    return pd.concat(all_detections, ignore_index=True)
