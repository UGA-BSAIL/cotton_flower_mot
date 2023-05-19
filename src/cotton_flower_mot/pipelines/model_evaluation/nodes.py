"""
Nodes for model evaluation pipeline.
"""


from typing import Dict, Callable, Iterable, List, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from functools import partial

from ..schemas import ModelInputs, ModelTargets
from ..schemas import ObjectTrackingFeatures as Otf
from .online_tracker import OnlineTracker, Track
from .tracking_video_maker import draw_tracks


def compute_tracks_for_clip(
    *,
    tracking_model: tf.keras.Model,
    detection_model: tf.keras.Model,
    clip_dataset: tf.data.Dataset,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Computes the tracks for a given sequence of clips.

    Args:
        tracking_model: The model to use for track computation.
        detection_model: The model to use for detection.
        clip_dataset: The dataset containing detections for each frame in the
            clip.

    Returns:
        Mapping of sequence IDs to tracks from that clip.

    """
    logger.info("Computing tracks for clip...")

    # Remove batch dimension and feed inputs one-at-a-time.
    clip_dataset = clip_dataset.unbatch()

    tracks_from_clips = {}
    current_sequence_id = -1

    tracker = None
    for inputs, _ in clip_dataset:
        # The two IDs should be identical anyway.
        sequence_id = int(inputs[ModelInputs.SEQUENCE_ID.value][0])
        if sequence_id != current_sequence_id:
            # Start of a new clip.
            logger.info("Starting tracking for clip {}.", sequence_id)

            if tracker is not None:
                tracks_from_clips[current_sequence_id] = tracker.tracks
            current_sequence_id = sequence_id
            tracker = OnlineTracker(
                tracking_model=tracking_model,
                detection_model=detection_model,
                death_window=10,
            )

        tracker.process_frame(
            frame=inputs[ModelInputs.DETECTIONS_FRAME.value].numpy(),
            detections=inputs[ModelInputs.DETECTION_GEOMETRY.value].numpy()[
                :, :4
            ],
        )
    # Add the last one.
    tracks_from_clips[current_sequence_id] = tracker.tracks

    # Serialize the tracks.
    for sequence, tracks in tracks_from_clips.items():
        tracks_from_clips[sequence] = [t.to_dict() for t in tracks]

    return tracks_from_clips


def compute_counts(
    *,
    tracks_from_clips: Dict[int, List[Dict[str, Any]]],
    annotations: pd.DataFrame,
) -> List:
    """
    Computes counts from the tracks and the overall counting accuracy.

    Args:
        tracks_from_clips: The extracted tracks from each clip.
        annotations: The raw annotations in Pandas form.

    Returns:
        A report about the count accuracy that is meant to be saved to a
        human-readable format.

    """
    clip_reports = []

    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    for sequence_id, tracks in tracks_from_clips.items():
        # Deserialize the tracks.
        tracks = [Track.from_dict(t) for t in tracks]

        # Calculate the ground-truth count.
        clip_annotations = annotations.iloc[annotations.index == sequence_id]
        gt_count = len(clip_annotations[Otf.OBJECT_ID.value].unique())

        # The predicted count is simply the number of tracks.
        predicted_count = len(tracks)
        count_error = gt_count - predicted_count

        clip_reports.append(
            dict(
                sequence_id=sequence_id,
                gt_count=gt_count,
                predicted_count=predicted_count,
                count_error=count_error,
            )
        )

    return clip_reports


def make_track_videos(
    *,
    tracks_from_clips: Dict[int, List[Dict[str, Any]]],
    clip_dataset: tf.data.Dataset,
) -> Dict[str, Callable[[], Iterable[np.ndarray]]]:
    """
    Creates track videos for all the tracks in a clip.

    Args:
        tracks_from_clips: The tracks that were found for each clip.
        clip_dataset: A dataset containing the input data for all the clips,
            sequentially.

    Yields:
        Each video, represented as an iterable of frames.

    """
    # Remove batching.
    clip_dataset = clip_dataset.unbatch()
    # Remove the targets and only keep the inputs.
    clip_dataset = clip_dataset.map(lambda i, _: i)

    def _draw_tracks(
        sequence_id_: int, tracks_: List[Dict[str, Any]]
    ) -> Iterable[np.ndarray]:
        # Deserialize the tracks.
        tracks_ = [Track.from_dict(t) for t in tracks_]

        logger.info(
            "Generating tracking video for sequence {}...", sequence_id_
        )

        # Filter the data to only this sequence.
        single_clip = clip_dataset.filter(
            lambda inputs: inputs[ModelInputs.SEQUENCE_ID.value][0]
            == sequence_id_
        )

        return draw_tracks(single_clip, tracks=tracks_)

    partitions = {}
    for sequence_id, tracks in tracks_from_clips.items():
        partitions[f"sequence_{sequence_id}"] = partial(
            _draw_tracks, sequence_id, tracks
        )

    return partitions
