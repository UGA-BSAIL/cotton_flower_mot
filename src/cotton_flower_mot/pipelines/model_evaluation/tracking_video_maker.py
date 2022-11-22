"""
Framework for creating tracking videos.
"""

import random
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from ..schemas import ModelInputs
from .online_tracker import Track

_TAG_FONT = ImageFont.truetype("fonts/VeraBd.ttf", 24)
"""
Font to use for bounding box tags.
"""


@lru_cache
def _color_for_track(track: Track) -> Tuple[int, int, int]:
    """
    Generates a unique color for a particular track.

    Args:
        track: The track to generate a color for.

    Returns:
        The generated color.

    """
    # Seed the random number generator with the track ID.
    random.seed(track.id)

    # Create a random color. We want it to be not very green (because the
    # background is pretty green), and relatively dark, so the label shows up
    # well.
    rgb = np.array(
        [
            random.randint(0, 255),
            random.randint(0, 128),
            random.randint(0, 255),
        ],
        dtype=np.float32,
    )

    brightness = np.sum(rgb)
    scale = brightness / 300
    # Keep a constant brightness.
    rgb *= scale

    return tuple(rgb.astype(int))


def _draw_text(
    artist: ImageDraw.ImageDraw,
    *,
    text: str,
    coordinates: Tuple[int, int],
    anchor: str = "la",
    color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draws text on an image, over a colored box.

    Args:
        artist: The `ImageDraw` object to draw with.
        text: The text to draw.
        coordinates: The coordinates to place the text at.
        anchor: The anchor type to use for the text.
        color: The background color to use. (The text itself will be white.)

    """
    # Find and draw the bounding box.
    text_bbox = artist.textbbox(
        coordinates, text, anchor=anchor, font=_TAG_FONT
    )
    artist.rectangle(text_bbox, fill=color)

    # Draw the text itself.
    artist.text(
        coordinates, text, fill=(255, 255, 255), anchor=anchor, font=_TAG_FONT
    )


def _draw_bounding_box(
    artist: ImageDraw.ImageDraw, *, track: Track, box: np.ndarray
) -> None:
    """
    Draws a bounding box.

    Args:
        artist: The `ImageDraw` object to draw with.
        track: The track that we are drawing a box for.
        box: The box to be drawn, in the form
            `[center_x, center_y, width, height]`.
    """
    # Convert the box to a form we can draw with.
    center = box[:2]
    size = box[2:]
    min_point = center - size // 2
    max_point = center + size // 2
    min_point = tuple(min_point)
    max_point = tuple(max_point)

    # Choose a color.
    color = _color_for_track(track)

    artist.rectangle((min_point, max_point), outline=color, width=5)

    # Draw a tag with the track ID.
    tag_pos = min_point
    _draw_text(
        artist,
        text=f"Track {track.id}",
        anchor="lb",
        color=color,
        coordinates=tag_pos,
    )


def draw_track_frame(
    frame: np.ndarray,
    *,
    frame_num: int,
    tracks: List[Track],
) -> np.ndarray:
    """
    Draws the tracks on a single frame.

    Args:
        frame: The frame to draw on. Will be modified in-place.
        frame_num: The frame number of this frame.
        tracks: The tracks to draw.

    Returns:
        The modified frame.

    """
    # Convert from OpenCV format to PIL.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(frame)

    # Determine the associated bounding box for all the tracks.
    for track in tracks:
        bounding_box = track.detection_for_frame(frame_num)
        if bounding_box is None:
            # No detection for this track at this frame.
            continue

        # Convert the bounding box to pixels.
        bounding_box *= np.array(
            [frame.width, frame.height, frame.width, frame.height]
        )
        # Because the image is flipped, we have to flip our bounding boxes.
        bounding_box[1] = frame.height - bounding_box[1]

        # Draw the bounding box.
        _draw_bounding_box(draw, track=track, box=bounding_box)

    return np.array(frame)


def draw_tracks(
    inputs: Iterable[Dict[str, tf.Tensor]], *, tracks: List[Track]
) -> Iterable[np.ndarray]:
    """
    Draws the tracks on top of a video.

    Args:
        inputs: Dictionary containing the input data, organized according
            to the keys in `ModelInputs`.
        tracks: The tracks to draw.

    Yields:
        Each frame, with the tracks drawn on it.

    """
    for frame_num, feature_dict in enumerate(inputs):
        frame = feature_dict[ModelInputs.DETECTIONS_FRAME.value].numpy()

        # Flip the frame, because the input data is upside-down.
        frame = cv2.flip(frame, 0)

        frame = draw_track_frame(frame, frame_num=frame_num, tracks=tracks)

        yield frame
