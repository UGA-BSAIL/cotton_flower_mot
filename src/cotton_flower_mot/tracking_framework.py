"""
Framework for online tracking.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .profiler import ProfilingManager
from .track import Track
from .model import DetectionModel
from .online_tracker import OnlineTracker


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


class OnlineTrackingFramework:
    """
    Performs online tracking using a given model.
    """

    def __init__(
        self,
        *,
        detection_model: DetectionModel,
        tracker: OnlineTracker,
        death_window: float = 1,
        confidence_threshold: float = 0.0,
        profile_manager: ProfilingManager = ProfilingManager(),
    ):
        """
        Args:
            detection_model: The model to use for tracking.
            tracker: The tracker to use.
            death_window: How many seconds we keep a track around after its
                last detection before we consider it dead.
            confidence_threshold: The confidence threshold to use for the
                detector.
            profile_manager: The profiling manager to use for profiling.

        """
        self.__detection_model = detection_model
        self.__tracker = tracker
        self.__death_window = death_window
        self.__confidence_threshold = confidence_threshold
        logger.info(
            f"Tracking configuration:\n"
            f"\tDeath window: {self.__death_window}\n"
            f"\tConfidence threshold: {self.__confidence_threshold}\n"
        )

        self._profiler = profile_manager

        # Stores the previous frame.
        self.__previous_frame = None
        self.__num_appearance_features = None
        # Stores the appearance features from the previous frame.
        self.__previous_appearance = None

        # Stores all the tracks that are currently active.
        self._active_tracks = set()
        # Stores all tracks that have been completed.
        self.__completed_tracks = []
        # Associates rows in the assignment matrix with corresponding tracks.
        self.__tracks_by_tracklet_index = {}

        # Counter for the current frame.
        self.__frame_num = 0

        # Average velocity of all completed tracks.
        self.__mean_velocity = np.zeros(2, dtype=np.float32)
        # Average velocity covariance of all completed tracks.
        self.__mean_velocity_cov = np.eye(2, dtype=np.float32)

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
        self._active_tracks.remove(track)
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
        sparse_assignment: np.array,
        detections: np.array,
        appearances: np.array,
        frame_time: float,
    ) -> None:
        """
        Updates the currently-active tracks with new detection information.

        Args:
            sparse_assignment: The sparse assignment matrix between the
                detections from the previous frame and the current one.
                Should have two columns, where the first is the index of the
                tracklet and the second the index of the corresponding
                detection.
            detections: The current detection boxes. Should have the shape
                `[num_detections, 4]`.
            appearances: The current appearance feature. Should have the shape
                `[num_detections, num_channels]`.
            frame_time: The current frame time.

        """
        # Figure out associations between tracklets and detections.
        dead_tracklets = []
        matched_track_indices = set()

        # Update matched tracks.
        for tracklet_index, detection_index in sparse_assignment:
            track = self.__tracks_by_tracklet_index[tracklet_index]
            matched_track_indices.add(tracklet_index)

            # Record the associated detection.
            track.add_new_detection(
                frame_num=self.__frame_num,
                detection=detections[detection_index],
                appearance_feature=appearances[detection_index],
                frame_time=frame_time,
            )

        # Update un-matched tracks.
        for tracklet_index, track in self.__tracks_by_tracklet_index.items():
            if tracklet_index in matched_track_indices:
                # We have a match for this one. Don't do anything.
                continue

            # It couldn't find a match for this tracklet.
            if frame_time - track.last_detection_time > self.__death_window:
                # Consider the tracklet dead.
                dead_tracklets.append(track)

            else:
                # Otherwise, extrapolate a new bounding box based on
                # previous track information.
                try:
                    with self._profiler.profile("motion_model"):
                        extrapolated_box = track.predict_future_box(frame_time)
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

        # Remove dead tracklets.
        logger.debug("Removing {} dead tracks.", len(dead_tracklets))
        for track in dead_tracklets:
            self.__retire_track(track)

    def __add_new_tracks(
        self,
        *,
        sparse_assignment: np.array,
        detections: np.array,
        appearances: np.array,
        frame_time: float,
    ) -> None:
        """
        Adds any new tracks to the set of active tracks.

        Args:
            sparse_assignment: The sparse assignment matrix between the
                detections from the previous frame and the current one.
                Should have two columns, where the first is the index of the
                tracklet and the second the index of the corresponding
                detection.
            detections: The current detections corresponding to this
                assignment matrix.
            appearances: The current appearance features corresponding to this
                assignment matrix.
            frame_time: The current frame time.

        """
        matched_detections = set(sparse_assignment[:, 1])

        for detection_index, (detection, appearance) in enumerate(
            zip(detections, appearances)
        ):
            if detection_index not in matched_detections:
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

                self._active_tracks.add(track)

    def __update_tracks(
        self,
        *,
        frame_time: float,
        sparse_assignment: np.array,
        detections: np.array,
        appearances: np.array,
    ) -> None:
        """
        Updates the current set of tracks based on the latest tracking result.

        Args:
            frame_time: The current frame time.
            sparse_assignment: The sparse assignment matrix between the
                detections from the previous frame and the current one.
                Should have two columns, where the first is the index of the
                tracklet and the second the index of the corresponding
                detection.
            detections: The current detection bounding boxes. Should have
                shape `[num_detections, 4]`.
            appearances: The current appearance features. Should have shape
                `[num_detections, num_channels]`.

        """
        logger.debug(sparse_assignment)

        with self._profiler.profile("update_active_tracks"):
            # Update the currently-active tracks.
            self.__update_active_tracks(
                sparse_assignment=sparse_assignment,
                detections=detections,
                appearances=appearances,
                frame_time=frame_time,
            )
        with self._profiler.profile("add_new_tracks"):
            self.__add_new_tracks(
                sparse_assignment=sparse_assignment,
                detections=detections,
                appearances=appearances,
                frame_time=frame_time,
            )

    def __update_saved_state(self, *, frame: np.ndarray) -> None:
        """
        Updates the saved frames and appearance features that
        will be used as the input tracks for the next frame.

        Args:
            frame: The current frame image. Should be an array of shape
                `[height, width, channels]`.

        """
        active_appearance = []
        self.__tracks_by_tracklet_index.clear()

        for i, track in enumerate(self._active_tracks):
            # Even if the appearance feature is older than the position
            # estimate, we'll still use it since it might be helpful.
            active_appearance.append(track.last_appearance)

            # Save the track object corresponding to this tracklet.
            self.__tracks_by_tracklet_index[i] = track

        self.__previous_frame = frame
        self.__previous_appearance = np.empty(
            (0, self.__num_appearance_features)
        )
        if len(active_appearance) > 0:
            self.__previous_appearance = np.stack(active_appearance, axis=0)

    def __create_tracking_inputs(
        self,
        *,
        detections: np.ndarray,
        appearance_features: np.ndarray,
        frame_time: float,
        restrict_to_tracklets: Optional[np.array] = None,
    ) -> Dict[str, np.array]:
        """
        Creates an input dictionary for the tracking model.

        Args:
            detections: The detections to add.
            appearance_features: The appearance features for the detections.
            frame_time: The timestamp of the current frame.
            restrict_to_tracklets: If provided, it will only include the
                tracklets with these specific indices.

        Returns:
            The input dictionary, which can be fed to `TrackingModel.track()`
            as keyword arguments.

        """
        detections = detections.astype(np.float32)
        appearance_features = appearance_features.astype(np.float32)
        previous_appearance = self.__previous_appearance.astype(np.float32)

        # Use the motion model to update the predicted geometry for the
        # current time step.
        previous_geometry = [None] * len(self._active_tracks)
        for i, track in self.__tracks_by_tracklet_index.items():
            previous_geometry[i] = track.predict_future_box(frame_time)
        if previous_geometry:
            previous_geometry = np.stack(previous_geometry, axis=0).astype(
                np.float32
            )
        else:
            previous_geometry = np.empty((0, 4))

        if restrict_to_tracklets is not None:
            previous_appearance = previous_appearance[restrict_to_tracklets]
            previous_geometry = previous_geometry[restrict_to_tracklets]

        return {
            "detections": detections,
            "tracklets": previous_geometry,
            "detections_appearance": appearance_features,
            "tracklets_appearance": previous_appearance,
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
        model_inputs = self.__create_tracking_inputs(
            detections=detection_geometry,
            appearance_features=appearance_features,
            frame_time=frame_time,
        )

        assignment = self.__tracker.associate(model_inputs)

        logger.debug("Got {} detections.", len(detection_geometry))
        # Remove the confidence, since we don't use that for tracking.
        detection_geometry = detection_geometry[:, :4]

        # Update the tracks.
        with self._profiler.profile("update_tracks"):
            self.__update_tracks(
                sparse_assignment=assignment,
                detections=detection_geometry,
                appearances=appearance_features,
                frame_time=frame_time,
            )

    def _do_detection(
        self, frame: np.array, *, _frame_time: float
    ) -> Tuple[np.array, np.array]:
        """
        Applies the detection model to the input frame.

        Args:
            frame: The frame to detect flowers in.
            _frame_time: The timestamp for this frame.

        Returns:
            - The box features, with shape `[num_boxes, 5]`
            - The appearance features, with shape
                `[num_boxes, channels]`.

        """
        # Apply the detector first.
        logger.debug("Applying detection model...")
        with self._profiler.profile("detection_model_full", warmup_iters=10):
            return self.__detection_model.detect(frame)

    def __match_frame_pair(
        self,
        *,
        frame_time: float,
        frame: np.array,
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
        with self._profiler.profile("detection", warmup_iters=10):
            detection_geometry, appearance_features = self._do_detection(
                frame, _frame_time=frame_time
            )

        with self._profiler.profile("filter_low_confidence"):
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
        with self._profiler.profile("update_saved_state"):
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
            with self._profiler.profile("match_frame_pair", warmup_iters=10):
                num_detections = self.__match_frame_pair(
                    frame=frame,
                    frame_time=frame_time,
                )

        self.__frame_num += 1

        return TrackingStats(
            num_detections=num_detections, num_tracks=len(self._active_tracks)
        )

    @property
    def tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that we have so far.

        """
        return self.__completed_tracks + list(self._active_tracks)

    @property
    def active_tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that are currently active.

        """
        return list(self._active_tracks)
