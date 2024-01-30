"""
Loads/stores a video from/to a sequence of frame images.
"""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from kedro.io import AbstractVersionedDataSet, Version

from ..frame_reader import FrameReader


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
        if not reader.isOpened():
            raise OSError(
                f"Could not open video {self._get_load_path()}. "
                f"Are you sure it exists?"
            )
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
