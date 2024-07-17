"""
Represents a single track.
"""


from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from .motion_model import MotionModel
from .running_median import RunningMedian


class Track:
    """
    Represents a single track.
    """

    _NEXT_ID = 1
    """
    Allows us to associate a unique ID with each track.
    """

    def __init__(self, mean_velocity: np.array, velocity_cov: np.array):
        """
        Args:
            mean_velocity: The mean velocity of all tracks so far, in the form
                `[v_x, v_y]`.
            velocity_cov: The covariance matrix of all the track velocities
                so far. Should be a 2x2 array.
        """
        # Maps frame numbers to detection bounding boxes.
        self.__frames_to_detections = {}
        # Maps frame numbers to anchor points.
        self.__frames_to_anchor_points = {}
        # Maps frame numbers to appearance features.
        self.__frames_to_appearance = {}
        # Maps frame numbers to whether this detection is real or extrapolated.
        self.__frame_has_detection = {}
        # Maps frame numbers to frame times.
        self.__frames_to_time = {}
        # Keeps track of the last frame we have a detection for.
        self.__latest_frame = -1
        # Keeps track of the last frame we have a motion estimation for.
        self.__latest_motion_frame = -1

        # Keep track of the median detection size.
        self.__median_width = RunningMedian()
        self.__median_height = RunningMedian()

        # Motion model to use.
        self.__motion_model: Optional[MotionModel] = None
        self.__mean_velocity = mean_velocity.copy()
        self.__velocity_cov = velocity_cov.copy()

        self.__id = Track._NEXT_ID
        Track._NEXT_ID += 1

    def __maybe_init_motion_model(
        self, frame_time: float, detection: np.array
    ) -> bool:
        """
        Initializes the motion model, if necessary.

        Args:
            frame_time: The time that the detection is from.
            detection: The detection bounding box, in the form
                `[center_x, center_y, width, height]`.

        Returns:
            True if the motion model was initialized, False if nothing needed
             to be done.

        """
        if self.__motion_model is None:
            initial_cov = np.eye(4, dtype=np.float32)
            initial_cov[2:, 2:] = self.__velocity_cov
            logger.debug(
                "Initializing motion model with box {}, vel {}, and cov {}.",
                detection,
                self.__mean_velocity,
                initial_cov,
            )

            self.__motion_model = MotionModel(
                initial_box=detection,
                initial_velocity=self.__mean_velocity,
                initial_cov=initial_cov,
                initial_time=frame_time,
            )

            return True
        return False

    def add_new_detection(
        self,
        *,
        frame_num: int,
        frame_time: float,
        detection: np.array,
        appearance_feature: Optional[np.array],
        is_extrapolated: bool = False,
    ) -> None:
        """
        Adds a new detection to the end of the track.

        Args:
            frame_num: The frame number that this detection is for.
            frame_time: The time that the detection is from.
            detection: The new detection to add, in the form
                `[center_x, center_y, width, height]`.
            appearance_feature: The appearance feature vector for this
                detection, in the form `[num_channels]`. It does not need to
                be provided if `is_extrapolated` is true.
            is_extrapolated: If true, marks this as an extrapolated detection
                instead of a "real" one.

        """
        if not is_extrapolated and appearance_feature is None:
            raise ValueError(
                "Appearance feature must be provided if box is "
                "not extrapolated."
            )

        self.__frames_to_detections[frame_num] = detection.copy()
        self.__frames_to_time[frame_num] = frame_time
        if appearance_feature is not None:
            self.__frames_to_appearance[frame_num] = appearance_feature.copy()
        self.__frame_has_detection[frame_num] = not is_extrapolated

        if not is_extrapolated:
            self.__latest_frame = max(self.__latest_frame, frame_num)

            # Update the motion model with the latest observation.
            if not self.__maybe_init_motion_model(frame_time, detection):
                self.__motion_model.add_observation(
                    detection, observed_time=frame_time
                )
            self.__frames_to_anchor_points[
                frame_num
            ] = self.__motion_model.anchor_point

            width, height = detection[2:]
            self.__median_width.add(width)
            self.__median_height.add(height)

        self.__latest_motion_frame = max(self.__latest_motion_frame, frame_num)

    @property
    def last_detection(self) -> Optional[np.ndarray]:
        """
        Returns:
            The bounding box of the current last detection in this track,
            or None if the track is empty.

        """
        return self.detection_for_frame(self.__latest_frame)

    @property
    def last_motion_estimate(self) -> Optional[np.ndarray]:
        """
        Returns:
            The bounding box of the most recent position of this object,
            as estimated by the motion model. Note that if we have an actual
            detection for this frame, the return value will be identical to
            that of `last_detection`.

        """
        return self.detection_for_frame(self.__latest_motion_frame)

    @property
    def last_velocity_estimate(self) -> Optional[np.ndarray]:
        """
        Returns:
            The most recent velocity estimate for this object from the motion
            model in the form `[vx, vy]` If there is no estimate, it will return
            None.

        """
        if self.__motion_model is not None:
            return self.__motion_model.state[2:]
        return None

    @property
    def last_appearance(self) -> Optional[np.ndarray]:
        """
        Returns:
            The most recent appearance feature for this track, or None if the
            track is empty.

        """
        return self.appearance_for_frame(self.__latest_frame)

    @property
    def last_detection_frame(self) -> Optional[int]:
        """
        Returns:
            The frame number at which this object was last detected.

        """
        if self.__latest_frame < 0:
            return None

        return self.__latest_frame

    @property
    def last_detection_time(self) -> Optional[int]:
        """
        Returns:
            The time at which this object was last detected.

        """
        if self.__latest_frame < 0:
            return None

        return self.__frames_to_time[self.__latest_frame]

    @property
    def last_tracked_frame(self) -> Optional[int]:
        """
        Returns:
            The frame number at which this object's position was last
            estimated. It might be extrapolated from the motion model instead
            of directly detected.
        """
        if self.__latest_motion_frame < 0:
            return None

        return self.__latest_motion_frame

    @property
    def first_detection_frame(self) -> Optional[int]:
        """
        Returns:
            The frame number at which this object was first detected.

        """
        if len(self.__frames_to_detections) == 0:
            return None
        return min(self.__frames_to_detections.keys())

    def detection_for_frame(self, frame_num: int) -> Optional[np.array]:
        """
        Gets the corresponding detection box for a particular frame,
        or None if we don't have a detection for that frame.

        Args:
            frame_num: The frame number.

        Returns:
            The index for that frame, or None if we don't have one.

        """
        if frame_num not in self.__frames_to_detections:
            return None
        return self.__frames_to_detections[frame_num].copy()

    def anchor_point_for_frame(self, frame_num: int) -> Optional[np.array]:
        """
        Gets the corresponding anchor point for a particular frame,
        or None if we don't have a detection for that frame.

        Args:
            frame_num: The frame number.

        Returns:
            The index for that frame, or None if we don't have one.

        """
        if frame_num not in self.__frames_to_anchor_points:
            return None
        return self.__frames_to_anchor_points[frame_num].copy()

    def appearance_for_frame(self, frame_num: int) -> Optional[np.array]:
        """
        Gets the corresponding appearance feature for a particular frame,
        or None if we don't have a detection for that frame.

        Args:
            frame_num: The frame number.

        Returns:
            The index for that frame, or None if we don't have one.

        """
        if frame_num not in self.__frames_to_appearance:
            return None
        return self.__frames_to_appearance[frame_num].copy()

    def all_detections(self) -> pd.DataFrame:
        """
        Gets all the detections for this track, as a DataFrame.

        Returns:
            All the detections, indexed by frame number.

        """
        return pd.DataFrame(
            index=self.__frames_to_detections.keys(),
            data=self.__frames_to_detections.values(),
            columns=["center_x", "center_y", "width", "height"],
        )

    def mean_velocity(self) -> np.array:
        """
        Computes the average velocity over the entire track.

        Returns:
            The average velocity, in the form `[x, y]`.

        """
        first_frame = self.first_detection_frame
        last_frame = self.last_detection_frame
        if first_frame == last_frame:
            # Velocity is undefined for track of length 1.
            return np.array([np.nan, np.nan])

        first_pos = self.__frames_to_detections[first_frame][:2]
        last_pos = self.__frames_to_detections[last_frame][:2]
        start_time = self.__frames_to_time[first_frame]
        end_time = self.__frames_to_time[last_frame]

        return (last_pos - first_pos) / (end_time - start_time)

    def velocity_cov(self) -> np.array:
        """
        Returns:
            The latest velocity covariance for this track, as a 2x2 matrix.

        """
        if self.__motion_model is None:
            raise ValueError(
                "Cannot use motion model before we have detections."
            )

        return self.__motion_model.cov[2:, 2:]

    def has_real_detection_for_frame(self, frame_num: int) -> bool:
        """
        Args:
            frame_num: The frame number to check at.

        Returns:
            True if there is an actual detection, False if there is no
            detection or only an extrapolated box.

        """
        return self.__frame_has_detection.get(frame_num, False)

    @property
    def id(self) -> int:
        """
        Returns:
            The unique ID associated with this track.

        """
        return self.__id

    def __len__(self) -> int:
        """
        Returns:
            The number of detections in the track.

        """
        return len(self.__frames_to_detections)

    def crosses_line(self, line_pos: float, horizontal: bool = True) -> bool:
        """
        Determines whether this track crosses a horizontal line.

        Args:
            line_pos: The height (if horizontal) or width (if vertical) of
                the line to check.
            horizontal: If true, use a horizontal line. Otherwise,
                use a vertical line.

        Returns:
            True if it crosses the line, false otherwise.

        """
        track_frames = list(self.__frames_to_detections.keys())
        track_frames.sort()

        was_before_line = None
        for frame_num in track_frames:
            box_center_x, box_center_y, _, _ = self.__frames_to_detections[
                frame_num
            ]

            if horizontal:
                is_before_line = box_center_y < line_pos
            else:
                is_before_line = box_center_x < line_pos

            if was_before_line is None:
                # No previous detection. Just save this and continue.
                was_before_line = is_before_line
            elif was_before_line != is_before_line:
                # It crossed the line.
                return True

        return False

    def predict_future_box(self, frame_time: float) -> np.array:
        """
        Extrapolates the trajectory of this object to a future time.

        Args:
            frame_time: The time that we are predicting the bounding
                box for.

        Returns:
            The extrapolated bounding box, of the form
            `[center_x, center_y, width, height]`.

        """
        if self.__motion_model is None:
            raise ValueError(
                "Cannot use motion model before we have detections."
            )
        state, _ = self.__motion_model.predict(frame_time)

        # Use the median size when it's occluded.
        median_size = np.array(
            [self.__median_width.median(), self.__median_height.median()]
        )
        return np.concatenate((state[:2], median_size))

    def to_dict(self) -> Dict[str, Any]:
        """
        Gets a dictionary representation of the track that can be easily
        serialized.

        Returns:
            A dictionary representing the track.

        """
        return dict(
            mean_velocity=self.__mean_velocity.tolist(),
            velocity_cov=self.__velocity_cov.tolist(),
            frames_to_detections={
                k: v.tolist() for k, v in self.__frames_to_detections.items()
            },
            frames_to_anchor_points={
                k: v.tolist()
                for k, v in self.__frames_to_anchor_points.items()
            },
            frame_has_detection=self.__frame_has_detection,
            frames_to_time=self.__frames_to_time,
            latest_frame=self.__latest_frame,
            track_id=self.__id,
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Track":
        """
        Creates a new track based on a serialized representation.

        Args:
            config: The serialized representation.

        Returns:
            The track that it created.

        """
        track = cls(
            mean_velocity=np.array(config["mean_velocity"]),
            velocity_cov=np.array(config["velocity_cov"]),
        )

        track.__frames_to_detections = {
            k: np.array(v) for k, v in config["frames_to_detections"].items()
        }
        track.__frames_to_anchor_points = {
            k: np.array(v)
            for k, v in config["frames_to_anchor_points"].items()
        }
        track.__frame_has_detection = config["frame_has_detection"]
        track.__frames_to_time = config["frames_to_time"]
        track.__latest_frame = config["latest_frame"]
        track.__id = config["track_id"]

        return track
