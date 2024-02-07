"""
Framework for creating tracking videos.
"""

import random
from functools import lru_cache
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from loguru import logger

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
    artist: ImageDraw.ImageDraw,
    *,
    track: Track,
    box: np.ndarray,
    is_detection: bool,
) -> None:
    """
    Draws a bounding box.

    Args:
        artist: The `ImageDraw` object to draw with.
        track: The track that we are drawing a box for.
        box: The box to be drawn, in the form
            `[center_x, center_y, width, height]`.
        is_detection: If true, this is a real detection, otherwise it is an
            extrapolated one.
    """
    # Convert the box to a form we can draw with.
    center = box[:2]
    size = box[2:]
    if size.min() < 0:
        logger.warning("Not drawing box with negative size.")
        return
    min_point = center - size // 2
    max_point = center + size // 2
    min_point = tuple(min_point)
    max_point = tuple(max_point)

    # Choose a color.
    color = _color_for_track(track)
    if not is_detection:
        # In this case, we want it to be translucent.
        color += (127,)

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


def _draw_counting_line(
    artist: ImageDraw.ImageDraw,
    *,
    pos: float,
    horizontal: bool,
    frame_width: int,
    frame_height: int,
) -> None:
    """
    Draws the counting line.

    Args:
        artist: The `ImageDraw` object to draw with.
        pos: The position of the line in the frame.
        horizontal: Whether the line is horizontal.
        frame_width: The width of the frame.
        frame_height: The height of the frame.

    """
    # Calculate the line coordinates.
    if horizontal:
        # Horizontal line
        pos_px = pos * frame_height
        line_coords = [(0, pos_px), (frame_width, pos_px)]
    else:
        # Vertical line
        pos_px = pos * frame_width
        line_coords = [(pos_px, 0), (pos_px, frame_height)]

    artist.line(line_coords, fill="red", width=3)


def draw_track_frame(
    frame: np.ndarray,
    *,
    frame_num: int,
    tracks: List[Track],
    line_pos: float,
    line_horizontal: bool,
) -> np.ndarray:
    """
    Draws the tracks on a single frame.

    Args:
        frame: The frame to draw on. Will be modified in-place.
        frame_num: The frame number of this frame.
        tracks: The tracks to draw.
        line_pos: The position of the counting line.
        line_horizontal: Whether the counting line is horizontal.

    Returns:
        The modified frame.

    """
    # Convert from OpenCV format to PIL.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame, mode="RGB")
    overlay = Image.new("RGBA", frame.size)
    draw = ImageDraw.Draw(overlay)

    # Determine the associated bounding box for all the tracks.
    for track in tracks:
        bounding_box = track.detection_for_frame(frame_num)
        if bounding_box is None:
            # No detection for this track at this frame.
            continue
        is_detection = track.has_real_detection_for_frame(frame_num)

        # Convert the bounding box to pixels.
        bounding_box *= np.array(
            [frame.width, frame.height, frame.width, frame.height]
        )
        # Because the image is flipped, we have to flip our bounding boxes.
        bounding_box[1] = frame.height - bounding_box[1]

        # Draw everything.
        _draw_bounding_box(
            draw, track=track, box=bounding_box, is_detection=is_detection
        )
        _draw_counting_line(
            draw,
            pos=line_pos,
            horizontal=line_horizontal,
            frame_width=frame.width,
            frame_height=frame.height,
        )

    frame = Image.alpha_composite(frame.convert("RGBA"), overlay)
    return np.array(frame.convert("RGB"))


def draw_tracks(
    clip_frames: Iterable[np.ndarray],
    *,
    tracks: List[Track],
    line_pos: float = 0.5,
    line_horizontal: bool = True,
) -> Iterable[np.ndarray]:
    """
    Draws the tracks on top of a video.

    Args:
        clip_frames: The video frames from the clip.
        tracks: The tracks to draw.
        line_pos: The position of the counting line.
        line_horizontal: Whether the counting line is horizontal.

    Yields:
        Each frame, with the tracks drawn on it.

    """
    for frame_num, frame in enumerate(clip_frames):
        # Flip the frame, because the input data is upside-down.
        frame = cv2.flip(frame, 0)

        frame = draw_track_frame(
            frame,
            frame_num=frame_num,
            tracks=tracks,
            line_pos=line_pos,
            line_horizontal=line_horizontal,
        )

        yield frame


def filter_short_tracks(
    tracks: Iterable[Track], min_length: int = 60
) -> Iterable[Track]:
    """
    Filters any tracks that span less than the minimum number of frames. Note
    that it will not include motion model extrapolation at the end of the
    track in this calculation.

    Args:
        tracks: The tracks to filter.
        min_length: The minimum length of a track.

    Returns:
        The filtered tracks.

    """
    for track in tracks:
        track_length = 0
        if track.last_detection_frame is not None:
            track_length = (
                track.last_detection_frame - track.first_detection_frame
            )

        if track_length >= min_length:
            yield track
