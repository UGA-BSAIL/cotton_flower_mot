"""
Wrapper around OpenCV captures that is easier to use.
"""


from functools import cached_property
from typing import Iterable, Tuple

import cv2
import numpy as np
from loguru import logger


class FrameReader:
    """
    A class that can be used to read frames starting at a particular spot.
    """

    def __init__(self, capture: cv2.VideoCapture, bgr_color: bool = True):
        """
        Args:
            capture: The `VideoCapture` that we are reading frames from.
            bgr_color: If true, it will load and save images in the BGR color
                space. Otherwise, it will load and save images in the RGB color
                space.

        """
        self.__capture = capture
        self.__bgr_color = bgr_color

    def __del__(self):
        self.__capture.release()

    @cached_property
    def num_frames(self) -> int:
        """
        Returns:
            The total number of frames in the video.

        """
        # Get the total number of frames.
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @cached_property
    def fps(self) -> float:
        """
        Returns:
            The reported FPS of the video.

        """
        return self.__capture.get(cv2.CAP_PROP_FPS)

    @cached_property
    def resolution(self) -> Tuple[int, int]:
        """
        Returns:
            The resolution of the video, in pixels, as (w, h).

        """
        first_frame = next(iter(self.read(0)))
        return tuple(first_frame.shape[:2][::-1])

    def read(self, start_frame: int) -> Iterable[np.ndarray]:
        """
        Iterates through the frames in a video.

        Args:
            start_frame: The frame to start reading at.

        Yields:
            Each frame that it read.

        """
        # Seek to the starting point.
        if start_frame >= self.num_frames:
            raise ValueError(
                f"Frame {start_frame} requested, but video has only"
                f" {self.num_frames} frames."
            )
        set_success = self.__capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        assert set_success

        # Read the frames.
        for _ in range(start_frame, self.num_frames):
            status, frame = self.__capture.read()
            if not status:
                logger.warning("Failed to read frame, skipping.")
                continue

            if not self.__bgr_color:
                # OpenCV works with BGR images, but we need RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame
