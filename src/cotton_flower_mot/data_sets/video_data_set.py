"""
Loads/stores a video from/to a sequence of frame images.
"""


from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from kedro.io import AbstractVersionedDataSet, Version
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


class VideoDataSet(AbstractVersionedDataSet):
    """
    Loads/stores a video from/to a sequence of frame images.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        version: Optional[Version] = None,
        codec: str = "mp4v",
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        bgr_color: bool = True,
    ):
        """
        Args:
            filepath: The path to the output video.
            version: The version information for the `DataSet`.
            codec: FourCC code to use for video encoding.
            fps: The FPS to use when writing the video.
            resolution: The output resolution of the video, in the form
                `(width, height)`.
            bgr_color: If true, it will load and save images in the BGR color
                space. Otherwise, it will load and save images in the RGB color
                space.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__codec = codec
        self.__fps = fps
        self.__resolution = resolution
        self.__bgr_color = bgr_color

    def _load(self) -> FrameReader:
        """
        Loads the video frame-by-frame.

        Returns:
            A function that can be used to read frames starting at a particular
            point, and reading up to a maximum number of frames.

        """
        reader = cv2.VideoCapture(self._get_load_path().as_posix())
        return FrameReader(reader, bgr_color=self.__bgr_color)

    def _save(self, data: Iterable[np.ndarray]) -> None:
        """
        Saves the video frame-by-frame.

        Args:
            data: An iterable of frames to save as a video.

        """
        # Make sure the save directory exists.
        save_path = Path(self._get_save_path())
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        codec = cv2.VideoWriter_fourcc(*self.__codec)
        writer = cv2.VideoWriter(
            self._get_save_path().as_posix(),
            codec,
            self.__fps,
            self.__resolution,
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        for frame in data:
            if self.__bgr_color:
                # OpenCV works with BGR images, but VideoWriter expects RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2][::-1] != self.__resolution:
                # Resize the frame to the correct shape.
                frame = cv2.resize(frame, self.__resolution)

            writer.write(frame)

        writer.release()

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            codec=self.__codec,
            fps=self.__fps,
            resolution=self.__resolution,
        )
