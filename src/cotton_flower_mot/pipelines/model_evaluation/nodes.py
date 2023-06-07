"""
Nodes for model evaluation pipeline.
"""


from typing import Dict, Callable, Iterable, List, Any, Tuple

import numpy as np
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
        )
    # Add the last one.
    tracks_from_clips[current_sequence_id] = tracker.tracks

    # Serialize the tracks.
    for sequence, tracks in tracks_from_clips.items():
        tracks_from_clips[sequence] = [t.to_dict() for t in tracks]

    return tracks_from_clips


def _get_line_for_sequence(
    counting_line_params: Dict[str, Any], sequence_id: int
) -> Tuple[float, bool]:
    """
    Gets the counting line for a particular sequence.

    Args:
        counting_line_params: The loaded counting line parameters.
        sequence_id: The ID of the sequence to get the line for.

    Returns:
        The line position, and whether it is horizontal or not.

    """
    line_positions = counting_line_params["line_positions"]
    sequences = counting_line_params["sequences"]
    track_camera = sequences[sequence_id]["camera"]
    track_pos = line_positions[track_camera]
    return track_pos["pos"], track_pos["horizontal"]


def compute_counts(
    *,
    tracks_from_clips: Dict[int, List[Dict[str, Any]]],
    counting_line_params: Dict[str, Any],
) -> List:
    """
    Computes counts from the tracks and the overall counting accuracy.

    Args:
        tracks_from_clips: The extracted tracks from each clip.
        counting_line_params: Parameters describing the counting line to use.

    Returns:
        A report about the count accuracy that is meant to be saved to a
        human-readable format.

    """
    clip_reports = []
    for sequence_id, tracks in tracks_from_clips.items():
        # Deserialize the tracks.
        tracks = [Track.from_dict(t) for t in tracks]

        # To determine the count, check for ones that cross the counting line.
        predicted_count = 0
        for track in tracks:
            line_pos, horizontal = _get_line_for_sequence(
                counting_line_params, sequence_id
            )
            if track.crosses_line(line_pos, horizontal=horizontal):
                predicted_count += 1

        clip_reports.append(
            dict(
                sequence_id=sequence_id,
                predicted_count=predicted_count,
            )
        )

    return clip_reports


def make_track_videos(
    *,
    tracks_from_clips: Dict[int, List[Dict[str, Any]]],
    clip_dataset: tf.data.Dataset,
    counting_line_params: Dict[str, Any],
) -> Dict[str, Callable[[], Iterable[np.ndarray]]]:
    """
    Creates track videos for all the tracks in a clip.

    Args:
        tracks_from_clips: The tracks that were found for each clip.
        clip_dataset: A dataset containing the input data for all the clips,
            sequentially.
        counting_line_params: Parameters describing the counting line to use.

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

        line_pos, horizontal = _get_line_for_sequence(
            counting_line_params, sequence_id_
        )
        return draw_tracks(
            single_clip,
            tracks=tracks_,
            line_pos=line_pos,
            line_horizontal=horizontal,
        )

    partitions = {}
    for sequence_id, tracks in tracks_from_clips.items():
        partitions[f"sequence_{sequence_id}"] = partial(
            _draw_tracks, sequence_id, tracks
        )

    return partitions
