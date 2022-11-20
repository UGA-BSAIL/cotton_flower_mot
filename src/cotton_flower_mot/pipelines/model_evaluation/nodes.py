"""
Nodes for model evaluation pipeline.
"""


from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

from ..schemas import ModelInputs
from ..schemas import ObjectTrackingFeatures as Otf
from .online_tracker import OnlineTracker, Track
from .tracking_video_maker import draw_tracks


def compute_tracks_for_clip(
    *, model: tf.keras.Model, clip_dataset: tf.data.Dataset
) -> Dict[int, List[Track]]:
    """
    Computes the tracks for a given sequence of clips.

    Args:
        model: The model to use for track computation.
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
            tracker = OnlineTracker(model)

        tracker.process_frame(
            frame=inputs[ModelInputs.DETECTIONS_FRAME.value].numpy(),
        )
    # Add the last one.
    tracks_from_clips[current_sequence_id] = tracker.tracks

    return tracks_from_clips


def compute_counts(
    *, tracks_from_clips: Dict[int, List[Track]], annotations: pd.DataFrame
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
    *, tracks_from_clips: Dict[int, List[Track]], clip_dataset: tf.data.Dataset
) -> Iterable[Iterable[np.ndarray]]:
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

    for sequence_id, tracks in tracks_from_clips.items():
        logger.info(
            "Generating tracking video for sequence {}...", sequence_id
        )

        # Filter the data to only this sequence.
        single_clip = clip_dataset.filter(
            lambda inputs: inputs[ModelInputs.SEQUENCE_ID.value][0]
            == sequence_id
        )

        yield draw_tracks(single_clip, tracks=tracks)
