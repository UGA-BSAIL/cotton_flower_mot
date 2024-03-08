"""
Performs tracking on a single video.
"""


import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import List, Any, Iterable

# Import sklearn here to avoid "cannot allocate memory in
# static TLS block" error on the Jetson.
import sklearn

from loguru import logger

from tqdm import tqdm

import av

import cv2

import pandas as pd

from src.cotton_flower_mot.tfrt_utils import (
    GraphFunc,
    get_func_from_saved_model,
    set_gpu_memory_limit,
)

set_gpu_memory_limit(256)

from src.cotton_flower_mot.frame_reader import FrameReader
from src.cotton_flower_mot.online_tracker import (
    Track,
    OnlineTracker,
)
from src.cotton_flower_mot.mot_challenge import track_to_mot_challenge
from src.cotton_flower_mot.tracking_video_maker import (
    draw_tracks,
    filter_short_tracks,
)

_DEATH_WINDOW_S = 1.0
"""
The number of seconds to use for the death window.
"""


def _configure_logging() -> None:
    """
    Configures logging for the application.

    """
    # Only print warnings and above to the console.
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    # Dump everything else to a file.
    logger.add("tracking_logs/tracking_{time}.log")


def _make_tracker(
    *,
    clip: FrameReader,
    tracking_model: GraphFunc,
    detection_model: GraphFunc,
    **kwargs: Any,
) -> OnlineTracker:
    """
    Creates an OnlineTracker instance for a given video.

    Args:
        clip: The video to create the tracker for.
        tracking_model: The model to use for tracking.
        detection_model: The model to use for detection.
        **kwargs: Will be forwarded to `OnlineTracker`.

    Returns:
        An OnlineTracker instance.

    """
    return OnlineTracker(
        detection_model=detection_model,
        tracking_model=tracking_model,
        death_window=_DEATH_WINDOW_S,
        **kwargs,
    )


def _compute_tracks_for_clip(
    *, clip: FrameReader, **kwargs: Any
) -> List[Track]:
    """
    Computes tracks for a single clip.

    Args:
        clip: The clip to compute tracks for.
        **kwargs: Will be forwarded to `_make_tracker()`.

    Returns:
        The computed tracks.

    """
    logger.info("Computing tracks for clip...")

    tracker = _make_tracker(
        clip=clip,
        **kwargs,
    )

    # We assume that all frames are evenly-spaced.
    frame_period = 1 / clip.fps
    frame_time = 0.0

    frames_progress = tqdm(clip.read(0), total=clip.num_frames)
    for frame in frames_progress:
        # Make sure it's the right size for the model.
        frame = cv2.resize(frame, (960, 540))

        stats = tracker.process_frame(frame, frame_time=frame_time)
        frame_time += frame_period
        frames_progress.set_postfix(asdict(stats))

    return tracker.tracks


def _save_video(
    *,
    clip: FrameReader,
    video_path: Path,
    tracks: List[Track],
) -> None:
    """
    Saves a version of the video with the tracks labeled on it.

    Args:
        clip: The video to save.
        video_path: The path to save the video to.
        tracks: The corresponding tracks for the video.

    """
    logger.info("Saving video to {}...", video_path)

    container = av.open(video_path.as_posix(), mode="w")
    stream = container.add_stream(
        "h264", rate=int(clip.fps), options={"crf": "23"}
    )
    stream.width = clip.resolution[0]
    stream.height = clip.resolution[1]
    stream.pix_fmt = "yuv420p"

    logger.debug("Drawing {} tracks...", len(tracks))
    for frame in draw_tracks(clip.read(0), tracks=tracks):
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush stream.
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _track_video(
    *,
    detection_model: Path,
    tracking_model: Path,
    video_path: Path,
    output_path: Path,
    bgr_color: bool = False,
    **kwargs: Any,
) -> None:
    """
    Performs tracking on a video.

    Args:
        detection_model: The path to the detection model.
        tracking_model: The path to the tracking model.
        video_path: The path to the video.
        output_path: Where to write the output tracking data file.
        bgr_color: Assume video uses BGR colorspace instead of RGB.
        **kwargs: Will be forwarded to `_compute_tracks_for_clip()`.

    """
    logger.info("Tracking from video {}...", video_path)

    # Load the models.
    detection_model, _ = get_func_from_saved_model(detection_model)
    tracking_model, __ = get_func_from_saved_model(tracking_model)

    # Load the video.
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise OSError(f"Failed to open video {video_path}.")
    clip = FrameReader(capture, bgr_color=bgr_color)
    logger.debug("Input resolution: {}x{}", *clip.resolution)

    # Compute the tracks.
    tracks = _compute_tracks_for_clip(
        clip=clip,
        detection_model=detection_model,
        tracking_model=tracking_model,
        **kwargs,
    )

    # Write the tracks to disk.
    tracks = list(
        filter_short_tracks(tracks, min_length=int(clip.fps * _DEATH_WINDOW_S))
    )
    tracks_tabular = [
        track_to_mot_challenge(track, clip.resolution) for track in tracks
    ]
    tracks_tabular = pd.concat(tracks_tabular)
    tracks_tabular.to_csv(output_path, index=False, header=False)

    # Save the video.
    output_video_path = output_path.with_name(f"{output_path.stem}_tracks.mp4")
    _save_video(clip=clip, video_path=output_video_path, tracks=tracks)


def _make_parser() -> argparse.ArgumentParser:
    """
    Returns:
        The parser to use for CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run the tracking system on a video."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="The saved model directory.",
    )
    parser.add_argument("video", type=Path, help="The path to the video file.")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the tracking result to.",
    )

    parser.add_argument(
        "-b",
        "--bgr-color",
        action="store_true",
        help="Assume video uses BGR color space instead of RGB.",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection model.",
    )
    parser.add_argument(
        "-i",
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold for fast association.",
    )
    parser.add_argument(
        "-g",
        "--gnn-only",
        action="store_true",
        help="Only use GNN model, with no pre-association stage.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()
    _configure_logging()

    output_path = cli_args.output or cli_args.video.with_suffix(".csv")
    _track_video(
        detection_model=cli_args.model_dir / "detection_model",
        tracking_model=cli_args.model_dir / "tracking_model",
        video_path=cli_args.video,
        output_path=output_path,
        bgr_color=cli_args.bgr_color,
        enable_two_stage_association=not cli_args.gnn_only,
        stage_one_iou_threshold=cli_args.iou,
        confidence_threshold=cli_args.conf,
    )


if __name__ == "__main__":
    main()
