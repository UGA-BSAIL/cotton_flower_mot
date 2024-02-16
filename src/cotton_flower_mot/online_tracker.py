"""
Framework for online tracking.
"""


from functools import singledispatch
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.linear_model import RANSACRegressor

from .schemas import ModelInputs, ModelTargets
from .assignment import (
    do_hard_assignment,
)
from .graph_utils import compute_pairwise_similarities
from .profiler import ProfilingManager
from .similarity_utils import compute_ious
from .motion_model import MotionModel


GraphFunc = Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
"""
Type alias for a function that runs the TF graph.
"""


@dataclass
class TrackingStats:
    """
    Represents tracking statistics.
    """

    num_detections: int
    """
    Number of detected objects on this iteration.
    """
    num_tracks: int
    """
    Number of objects currently being tracked.
    """


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
            initial_state = np.concatenate(
                (detection[:2], self.__mean_velocity)
            )
            initial_cov = np.eye(4, dtype=np.float32)
            initial_cov[2:, 2:] = self.__velocity_cov
            logger.debug(
                "Initializing motion model with state {} and cov {}.",
                initial_state,
                initial_cov,
            )

            self.__motion_model = MotionModel(
                initial_state=initial_state,
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
                    detection[:2], observed_time=frame_time
                )

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
        return self.__frames_to_detections[frame_num]

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
        return self.__frames_to_appearance[frame_num]

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

        # Assume that the size stays the same.
        return np.concatenate((state[:2], self.last_detection[2:]))

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
            k: np.array(v) for k, v in config["frames_to_detections"]
        }
        track.__frame_has_detection = config["frame_has_detection"]
        track.__frames_to_time = config["frames_to_time"]
        track.__latest_frame = config["latest_frame"]
        track.__id = config["track_id"]

        return track


@singledispatch
def _adapt_detection_model(model: GraphFunc) -> GraphFunc:
    """
    Adapts the model to use standardized inputs and produce standardized
    outputs.

    Args:
        model: The model to adapt.

    Returns:
        The adapted model.

    """

    def _adapted_model(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # This model expects float inputs...
        input_frame = inputs[ModelInputs.DETECTIONS_FRAME.value]
        inputs[ModelInputs.DETECTIONS_FRAME.value] = tf.cast(
            input_frame, tf.float32
        )

        appearance, geometry, _ = model(**inputs).values()
        return dict(appearance=appearance, geometry=geometry)

    return _adapted_model


@_adapt_detection_model.register
def _(model: tf.keras.Model) -> GraphFunc:
    def _adapted_model(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        geometry, appearance = model(inputs, training=False)
        return dict(appearance=appearance, geometry=geometry)

    return _adapted_model


@singledispatch
def _adapt_tracking_model(model: GraphFunc) -> GraphFunc:
    """
    Adapts the model to use standardized inputs and produce standardized
    outputs.

    Args:
        model: The model to adapt.

    Returns:
        The adapted model.

    """

    def _adapted_model(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Add the row length inputs.
        detection_geometry = inputs[ModelInputs.DETECTION_GEOMETRY.value]
        tracklet_geometry = inputs[ModelInputs.TRACKLET_GEOMETRY.value]
        detection_row_lengths = tf.reshape(
            tf.shape(detection_geometry)[1], (1, 1)
        )
        tracklet_row_lengths = tf.reshape(
            tf.shape(tracklet_geometry)[1], (1, 1)
        )
        flat_inputs = {
            "detection_appearance_row_lengths": detection_row_lengths,
            "detection_geometry_row_lengths": detection_row_lengths,
            "tracklet_appearance_row_lengths": tracklet_row_lengths,
            "tracklet_geometry_row_lengths": tracklet_row_lengths,
            "detection_appearance_flat": inputs[
                ModelInputs.DETECTION_APPEARANCE.value
            ],
            "detection_geometry_flat": detection_geometry,
            "tracklet_appearance_flat": inputs[
                ModelInputs.TRACKLET_APPEARANCE.value
            ],
            "tracklet_geometry_flat": tracklet_geometry,
        }

        sinkhorn, _ = model(**flat_inputs).values()
        return {ModelTargets.SINKHORN.value: sinkhorn}

    return _adapted_model


@_adapt_tracking_model.register
def _(model: tf.keras.Model) -> GraphFunc:
    def _adapted_model(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Use ragged inputs.
        for input_name, input_value in inputs.items():
            inputs[input_name] = tf.RaggedTensor.from_tensor(input_value)

        sinkhorn, _ = model(inputs, training=False)

        return {ModelTargets.SINKHORN.value: sinkhorn}


class OnlineTracker:
    """
    Performs online tracking using a given model.
    """

    def __init__(
        self,
        *,
        tracking_model: Union[GraphFunc, tf.keras.Model],
        detection_model: Union[GraphFunc, tf.keras.Model],
        death_window: float = 1,
        confidence_threshold: float = 0.0,
        stage_one_iou_threshold: float = 0.5,
        enable_two_stage_association: bool = True,
    ):
        """
        Args:
            tracking_model: The model to use for tracking.
            detection_model: The model to use for tracking.
            death_window: How many seconds we keep a track around after its
                last detection before we consider it dead.
            confidence_threshold: The confidence threshold to use for the
                detector.
            stage_one_iou_threshold: The IOU threshold to use for the
                first-stage fast association step.
            enable_two_stage_association: If true, it will attempt a fast
                IOU-based association process and only fall back on the GNN
                if that fails.

        """
        self.__tracking_model = _adapt_tracking_model(tracking_model)
        self.__detection_model = _adapt_detection_model(detection_model)
        self.__death_window = death_window
        self.__confidence_threshold = confidence_threshold
        self.__iou_threshold = tf.constant(stage_one_iou_threshold)
        self.__enable_fast_association = enable_two_stage_association
        logger.info(
            f"Tracker configuration:\n"
            f"\tTwo-stage association: {self.__enable_fast_association}\n"
            f"\tDeath window: {self.__death_window}\n"
            f"\tConfidence threshold: {self.__confidence_threshold}\n"
            f"\tIOU threshold: {self.__iou_threshold}"
        )

        # Stores the previous frame.
        self.__previous_frame = None
        # Stores the detection geometry from the previous frame.
        self.__previous_geometry = np.empty((0, 4), dtype=np.float32)
        self.__num_appearance_features = None
        # Stores the appearance features from the previous frame.
        self.__previous_appearance = None

        # Stores all the tracks that are currently active.
        self.__active_tracks = set()
        # Stores all tracks that have been completed.
        self.__completed_tracks = []
        # Associates rows in __previous_detections and __previous_geometry
        # with corresponding tracks.
        self.__tracks_by_tracklet_index = {}

        # Counter for the current frame.
        self.__frame_num = 0

        # Average velocity of all completed tracks.
        self.__mean_velocity = np.zeros(2, dtype=np.float32)
        # Average velocity covariance of all completed tracks.
        self.__mean_velocity_cov = np.eye(2, dtype=np.float32)

        # Internal profiler to use.
        self.__profiler = ProfilingManager()

    def __maybe_init_state(self, *, frame: np.ndarray) -> bool:
        """
        Initializes the previous detection state from the current detections
        if necessary.

        Args:
            frame: The current frame image.

        Returns:
            True if the state was initialized with the detections.

        """
        if self.__previous_frame is None:
            logger.debug(
                "Initializing tracker state.",
            )
            self.__previous_frame = frame

            return True
        return False

    def __maybe_init_appearance(self, appearance_features: np.array) -> None:
        """
        Initialize the saved appearance features if necessary.

        Args:
            appearance_features: The current appearance features.

        """
        if self.__previous_appearance is None:
            self.__num_appearance_features = appearance_features.shape[-1]
            logger.debug(
                "Initializing with {} appearance features.",
                self.__num_appearance_features,
            )
            # Stores the appearance features from the previous frame.
            self.__previous_appearance = np.empty(
                (0, self.__num_appearance_features), dtype=np.float32
            )

    def __retire_track(self, track: Track) -> None:
        """
        Finalizes a dead track, performing all necessary bookkeeping.

        Args:
            track: The dead track.

        """
        num_completed = len(self.__completed_tracks)
        self.__active_tracks.remove(track)
        self.__completed_tracks.append(track)

        # Update the running velocity statistics.
        track_vel = track.mean_velocity()
        if np.any(np.isnan(track_vel)):
            # The track only has one detection, so we can't get velocity.
            logger.debug("Skipping vel update for 1-length track.")
            return

        if num_completed == 0:
            self.__mean_velocity = track.mean_velocity()
            self.__mean_velocity_cov = track.velocity_cov()
        else:
            average_ratio = num_completed / (num_completed + 1)
            self.__mean_velocity += track.mean_velocity() / num_completed
            self.__mean_velocity *= average_ratio
            self.__mean_velocity_cov += track.velocity_cov() / num_completed
            self.__mean_velocity_cov *= average_ratio
        logger.debug(
            "Mean velocity: {}, Cov: {}",
            self.__mean_velocity,
            self.__mean_velocity_cov,
        )

    def __update_active_tracks(
        self,
        *,
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
        frame_time: float,
    ) -> None:
        """
        Updates the currently-active tracks with new detection information.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.
            detections: The current detection boxes. Should have the shape
                `[num_detections, 4]`.
            appearances: The current appearance feature. Should have the shape
                `[num_detections, num_channels]`.
            frame_time: The current frame time.

        """
        # Figure out associations between tracklets and detections.
        dead_tracklets = []
        for tracklet_index, track in self.__tracks_by_tracklet_index.items():
            tracklet_row = assignment_matrix[tracklet_index]
            if not np.any(tracklet_row):
                # It couldn't find a match for this tracklet.
                if (
                    frame_time - track.last_detection_time
                    > self.__death_window
                ):
                    # Consider the tracklet dead.
                    dead_tracklets.append(track)

                else:
                    # Otherwise, extrapolate a new bounding box based on
                    # previous track information.
                    try:
                        with self.__profiler.profile("motion_model"):
                            extrapolated_box = track.predict_future_box(
                                frame_time
                            )
                    except ValueError:
                        logger.debug(
                            "Not extrapolating track because there "
                            "are too few detections."
                        )
                        continue
                    track.add_new_detection(
                        frame_num=self.__frame_num,
                        frame_time=frame_time,
                        detection=extrapolated_box,
                        appearance_feature=None,
                        is_extrapolated=True,
                    )

            else:
                # Find the associated detection.
                new_detection_index = np.argmax(tracklet_row)
                track.add_new_detection(
                    frame_num=self.__frame_num,
                    detection=detections[new_detection_index],
                    appearance_feature=appearances[new_detection_index],
                    frame_time=frame_time,
                )

        # Remove dead tracklets.
        logger.debug("Removing {} dead tracks.", len(dead_tracklets))
        for track in dead_tracklets:
            self.__retire_track(track)

    def __add_new_tracks(
        self,
        *,
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
        frame_time: float,
    ) -> None:
        """
        Adds any new tracks to the set of active tracks.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.
            detections: The current detections corresponding to this
                assignment matrix.
            appearances: The current appearance features corresponding to this
                assignment matrix.
            frame_time: The current frame time.

        """
        for detection_index, (detection, appearance) in enumerate(
            zip(detections, appearances)
        ):
            detection_col = assignment_matrix[:, detection_index]
            if not np.any(detection_col):
                # There is no associated tracklet with this detection,
                # so it represents a new track.
                track = Track(
                    mean_velocity=self.__mean_velocity,
                    velocity_cov=self.__mean_velocity_cov,
                )
                logger.debug("Adding new track from detection {}.", detection)
                track.add_new_detection(
                    frame_num=self.__frame_num,
                    detection=detection,
                    appearance_feature=appearance,
                    frame_time=frame_time,
                )

                self.__active_tracks.add(track)

    def __sinkhorn_to_assigment(
        self,
        sinkhorn_matrix: np.array,
        *,
        detections: np.array,
    ) -> np.array:
        """
        Converts a sinkhorn matrix to a hard assignment matrix.

        Args:
            sinkhorn_matrix: The sinkhorn matrix between the detections from
                the previous frame and the current one. Should have a shape of
                `[num_detections * num_tracklets]`.
            detections: The current detection bounding boxes. Should have
                shape `[num_detections, 4]`.

        """
        # Un-flatten the sinkhorn matrix.
        num_tracklets = len(self.__previous_geometry)
        num_detections = len(detections)
        sinkhorn_matrix = np.reshape(
            sinkhorn_matrix, (num_tracklets + 1, num_detections + 1)
        )
        logger.debug(sinkhorn_matrix)

        assignment = do_hard_assignment(sinkhorn_matrix).numpy()
        logger.debug("Expanding assignment matrix to {}.", assignment.shape)

        return assignment

    def __update_tracks(
        self,
        *,
        frame_time: float,
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
    ) -> None:
        """
        Updates the current set of tracks based on the latest tracking result.

        Args:
            frame_time: The current frame time.
            assignment_matrix: The boolean assignment matrix between the
                detections from the previous frame and the current one.
                Should have a shape of `[num_tracklets, num_detections]`.
            detections: The current detection bounding boxes. Should have
                shape `[num_detections, 4]`.
            appearances: The current appearance features. Should have shape
                `[num_detections, num_channels]`.

        """
        logger.debug(assignment_matrix)

        with self.__profiler.profile("update_active_tracks"):
            # Update the currently-active tracks.
            self.__update_active_tracks(
                assignment_matrix=assignment_matrix,
                detections=detections,
                appearances=appearances,
                frame_time=frame_time,
            )
        with self.__profiler.profile("add_new_tracks"):
            self.__add_new_tracks(
                assignment_matrix=assignment_matrix,
                detections=detections,
                appearances=appearances,
                frame_time=frame_time,
            )

    def __update_saved_state(self, *, frame: np.ndarray) -> None:
        """
        Updates the saved frames, detections and appearance features that
        will be used as the input tracks for the next frame.

        Args:
            frame: The current frame image. Should be an array of shape
                `[height, width, channels]`.

        """
        active_geometry = []
        active_appearance = []
        self.__tracks_by_tracklet_index.clear()

        for i, track in enumerate(self.__active_tracks):
            active_geometry.append(track.last_motion_estimate)
            # Even if the appearance feature is older than the position
            # estimate, we'll still use it since it might be helpful.
            active_appearance.append(track.last_appearance)

            # Save the track object corresponding to this tracklet.
            self.__tracks_by_tracklet_index[i] = track

        self.__previous_frame = frame
        self.__previous_geometry = np.empty((0, 4))
        if len(active_geometry) > 0:
            self.__previous_geometry = np.stack(active_geometry, axis=0)

        self.__previous_appearance = np.empty(
            (0, self.__num_appearance_features)
        )
        if len(active_appearance) > 0:
            self.__previous_appearance = np.stack(active_appearance, axis=0)

    @staticmethod
    def __create_detection_inputs(
        *, frame: np.ndarray
    ) -> Dict[str, Union[tf.RaggedTensor, tf.Tensor]]:
        """
        Creates an input dictionary for the detection model.

        Args:
            frame: The current frame image. Should be an array of shape
                `[width, height, num_channels]`.

        """
        # Expand dimensions since the model expects a batch.
        frame = np.expand_dims(frame, axis=0)
        return {
            ModelInputs.DETECTIONS_FRAME.value: frame,
        }

    def __create_tracking_inputs(
        self, *, detections: np.ndarray, appearance_features: np.ndarray
    ) -> Dict[str, Union[tf.RaggedTensor, tf.Tensor]]:
        """
        Creates an input dictionary for the tracking model..

        Args:
            detections: The detections to add.
            appearance_features: The appearance features for the detections.

        """
        # Expand dimensions since the model expects a batch.
        detections = np.expand_dims(detections, axis=0).astype(np.float32)
        appearance_features = np.expand_dims(
            appearance_features, axis=0
        ).astype(np.float32)
        previous_geometry = np.expand_dims(
            self.__previous_geometry, axis=0
        ).astype(np.float32)
        previous_appearance = np.expand_dims(
            self.__previous_appearance, axis=0
        ).astype(np.float32)

        return {
            ModelInputs.DETECTION_GEOMETRY.value: detections,
            ModelInputs.TRACKLET_GEOMETRY.value: previous_geometry,
            ModelInputs.DETECTION_APPEARANCE.value: appearance_features,
            ModelInputs.TRACKLET_APPEARANCE.value: previous_appearance,
        }

    def __filter_low_confidence_detections(
        self, geometry: np.array, appearance: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Filters out low-confidence detections.

        Args:
            geometry: The detection geometry, with confidence.
            appearance: The corresponding appearance features.

        Returns:
            The corresponding filtered geometry and appearance features. Will
            also remove the extra confidence values.

        """
        confidence = geometry[:, 4]
        mask = confidence >= self.__confidence_threshold
        return geometry[mask][:, :4], appearance[mask]

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def _fast_association_impl(
        self,
        geometry: tf.Tensor,
        previous_geometry: tf.Tensor,
        iou_threshold: tf.Tensor,
    ) -> tf.Tensor:
        """
        Implementation of the fast association routine.

        Args:
            geometry: The current bounding boxes.
            previous_geometry: The previous bounding boxes.
            iou_threshold: The IOU threshold to use for determining matches.

        Returns:
            The boolean assignment matrix of shape `[num_tracklets,
            num_detections], if association succeeded, otherwise an empty
            tensor.

        """
        # First, compute IOUs between all tracklets and all detections.
        geometry = tf.expand_dims(geometry, axis=0)
        previous_geometry = tf.expand_dims(
            previous_geometry,
            axis=0,
        )
        pairwise_ious = compute_pairwise_similarities(
            compute_ious,
            left_features=previous_geometry,
            right_features=geometry,
        )[0]

        valid_matches = tf.greater(pairwise_ious, iou_threshold)
        valid_matches_int = tf.cast(valid_matches, tf.int32)
        num_tracklets_matches = tf.reduce_sum(valid_matches_int, axis=0)
        num_detections_matches = tf.reduce_sum(valid_matches_int, axis=1)

        invalid_criterion = tf.logical_or(
            # To meet the criteria for a valid association, each detection must
            # have AT MOST ONE plausible association with a tracklet.
            tf.logical_or(
                tf.reduce_any(num_tracklets_matches > 1),
                tf.reduce_any(num_detections_matches > 1),
            ),
            # Also, we can't have any tracklet/detection pairs that could have
            # been matched but weren't.
            tf.less(
                tf.reduce_sum(valid_matches_int),
                tf.reduce_min(tf.shape(valid_matches_int)),
            ),
        )

        empty_tensor = tf.constant([], dtype=tf.bool)
        return tf.cond(
            invalid_criterion,
            lambda: empty_tensor,
            lambda: valid_matches,
        )

    def __do_fast_association(self, geometry: np.array) -> Optional[np.array]:
        """
        Performs an initial fast attempt at association based on the bounding
        box IOUs.

        Args:
            geometry: The current bounding boxes.

        Returns:
            The boolean assignment matrix of shape `[num_tracklets,
            num_detections], if association succeeded, otherwise an empty
            array.

        """
        with self.__profiler.profile("fast_association", warmup_iters=10):
            geometry = tf.convert_to_tensor(geometry, dtype=tf.float32)
            previous_geometry = tf.convert_to_tensor(
                self.__previous_geometry, dtype=tf.float32
            )

            assignment = self._fast_association_impl(
                geometry, previous_geometry, self.__iou_threshold
            ).numpy()

            if len(assignment) == 0:
                # The association didn't work.
                return None
            return assignment

    def __do_association(
        self,
        *,
        frame_time: float,
        detection_geometry: np.array,
        appearance_features: np.array,
    ) -> None:
        """
        Performs the association step of the tracking pipeline.

        Args:
            frame_time: The current frame time.
            detection_geometry: The detection bounding boxes.
            appearance_features: The detection appearance features.

        """
        num_tracklets = self.__previous_geometry.shape[0]
        num_detections = detection_geometry.shape[0]

        if num_tracklets == 0 or num_detections == 0:
            # Don't bother running the tracker.
            logger.debug("No tracks or no detections, not running tracker.")
            assignment = np.zeros((num_tracklets, num_detections), dtype=bool)
        elif (
            not self.__enable_fast_association
            or (assignment := self.__do_fast_association(detection_geometry))
            is None
        ):
            # Fast association failed.
            logger.debug("Falling back on slow association...")
            model_inputs = self.__create_tracking_inputs(
                detections=detection_geometry,
                appearance_features=appearance_features,
            )
            with self.__profiler.profile("slow_association", warmup_iters=10):
                model_outputs = self.__tracking_model(model_inputs)
                sinkhorn = model_outputs[ModelTargets.SINKHORN.value][
                    0
                ].numpy()
                assignment = self.__sinkhorn_to_assigment(
                    sinkhorn, detections=detection_geometry
                )

        logger.debug("Got {} detections.", len(detection_geometry))
        # Remove the confidence, since we don't use that for tracking.
        detection_geometry = detection_geometry[:, :4]

        # Update the tracks.
        with self.__profiler.profile("update_tracks"):
            self.__update_tracks(
                assignment_matrix=assignment,
                detections=detection_geometry,
                appearances=appearance_features,
                frame_time=frame_time,
            )

    def __match_frame_pair(
        self,
        *,
        frame_time: float,
        frame: np.ndarray,
    ) -> int:
        """
        Computes the assignment matrix between the current state and new
        detections, and updates the state.

        Args:
            frame: The current frame. Should be an array of shape
                `[height, width, channels]`.
            frame_time: The current frame time.

        Returns:
            The number of detected objects in this frame.

        """
        with self.__profiler.profile("create_detection_inputs"):
            model_inputs = self.__create_detection_inputs(frame=frame)
        # Apply the detector first.
        logger.debug("Applying detection model...")
        with self.__profiler.profile("detection_model", warmup_iters=10):
            detections = self.__detection_model(model_inputs)
        detection_geometry = detections["geometry"][0].numpy()
        appearance_features = detections["appearance"][0].numpy()
        with self.__profiler.profile("filter_low_confidence"):
            (
                detection_geometry,
                appearance_features,
            ) = self.__filter_low_confidence_detections(
                detection_geometry, appearance_features
            )
        self.__maybe_init_appearance(appearance_features)

        self.__do_association(
            detection_geometry=detection_geometry,
            appearance_features=appearance_features,
            frame_time=frame_time,
        )

        # Update the state.
        with self.__profiler.profile("update_saved_state"):
            self.__update_saved_state(frame=frame)

        return len(detection_geometry)

    def process_frame(
        self, frame: np.array, *, frame_time: float
    ) -> TrackingStats:
        """
        Use the tracker to process a new frame. It will detect objects in the
        new frame and update the current tracks.

        Args:
            frame: The original image frame from the video. Should be an
                array of shape `[height, width, channels]`.
            frame_time: The time at which this frame was captured.

        """
        num_detections = 0
        if not self.__maybe_init_state(frame=frame):
            with self.__profiler.profile("match_frame_pair", warmup_iters=10):
                num_detections = self.__match_frame_pair(
                    frame=frame,
                    frame_time=frame_time,
                )

        self.__frame_num += 1

        return TrackingStats(
            num_detections=num_detections, num_tracks=len(self.__active_tracks)
        )

    @property
    def tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that we have so far.

        """
        return self.__completed_tracks + list(self.__active_tracks)
