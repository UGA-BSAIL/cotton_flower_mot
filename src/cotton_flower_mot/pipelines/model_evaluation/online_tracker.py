"""
Framework for online tracking.
"""


from typing import Dict, Iterable, List, Optional, Union, Any

import numpy as np
import tensorflow as tf
from loguru import logger

from ..schemas import ModelInputs
from ..assignment import do_hard_assignment, add_births_and_deaths


class Track:
    """
    Represents a single track.
    """

    _NEXT_ID = 1
    """
    Allows us to associate a unique ID with each track.
    """

    def __init__(self, indices: Iterable[int] = ()):
        """
        Args:
            indices: Initial indices into the detections array for each
                frame that form the track.
        """
        # Maps frame numbers to detection bounding boxes.
        self.__frames_to_detections = {f: i for f, i in enumerate(indices)}
        # Keeps track of the last frame we have a detection for.
        self.__latest_frame = -1

        self.__id = Track._NEXT_ID
        Track._NEXT_ID += 1

    def add_new_detection(
        self, *, frame_num: int, detection: np.ndarray
    ) -> None:
        """
        Adds a new detection to the end of the track.

        Args:
            frame_num: The frame number that this detection is for.
            detection: The new detection to add, in the form
                `[center_x, center_y, width, height]`.

        """
        self.__frames_to_detections[frame_num] = detection.tolist()
        self.__latest_frame = max(self.__latest_frame, frame_num)

    @property
    def last_detection(self) -> Optional[np.ndarray]:
        """
        Returns:
            The bounding box of the current last detection in this track,
            or None if the track is empty.

        """
        return self.detection_for_frame(self.__latest_frame)

    @property
    def last_detection_frame(self) -> Optional[int]:
        """
        Returns:
            The frame number at which this object was last detected.

        """
        return self.__latest_frame

    def detection_for_frame(self, frame_num: int) -> Optional[np.ndarray]:
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
        return np.array(self.__frames_to_detections[frame_num])

    @property
    def id(self) -> int:
        """
        Returns:
            The unique ID associated with this track.

        """
        return self.__id

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Gets a dictionary representation of the track that can be easily
        serialized.

        Returns:
            A dictionary representing the track.

        """
        return dict(
            frames_to_detections=self.__frames_to_detections,
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
        track = cls()

        track.__frames_to_detections = config["frames_to_detections"]
        track.__latest_frame = config["latest_frame"]
        track.__id = config["track_id"]

        return track


class OnlineTracker:
    """
    Performs online tracking using a given model.
    """

    def __init__(
        self,
        *,
        tracking_model: tf.keras.Model,
        detection_model: tf.keras.Model,
        death_window: int = 10,
    ):
        """
        Args:
            tracking_model: The model to use for tracking.
            detection_model: The model to use for tracking.
            death_window: How many consecutive frames we have to not observe
                a tracklet for before we consider it dead.

        """
        self.__tracking_model = tracking_model
        self.__detection_model = detection_model
        self.__death_window = death_window

        # Stores the previous frame.
        self.__previous_frame = None
        # Stores the detection geometry from the previous frame.
        self.__previous_geometry = np.empty((0, 4), dtype=np.float32)

        # Stores all the tracks that are currently active.
        self.__active_tracks = set()
        # Stores all tracks that have been completed.
        self.__completed_tracks = []
        # Associates rows in __previous_detections and __previous_geometry
        # with corresponding tracks.
        self.__tracks_by_tracklet_index = {}

        # Counter for the current frame.
        self.__frame_num = 0

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

    def __update_active_tracks(
        self, *, assignment_matrix: np.ndarray, detections: np.ndarray
    ) -> None:
        """
        Updates the currently-active tracks with new detection information.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.
            detections: The current detection boxes. Should have the shape
                `[num_detections, 4]`.

        """
        # Figure out associations between tracklets and detections.
        dead_tracklets = []
        for tracklet_index, track in self.__tracks_by_tracklet_index.items():
            tracklet_row = assignment_matrix[tracklet_index]
            if not np.any(tracklet_row):
                # It couldn't find a match for this tracklet.
                if (
                    self.__frame_num - track.last_detection_frame
                    > self.__death_window
                ):
                    # Consider the tracklet dead.
                    dead_tracklets.append(track)

            else:
                # Find the associated detection.
                new_detection_index = np.argmax(
                    assignment_matrix[tracklet_index]
                )
                track.add_new_detection(
                    frame_num=self.__frame_num,
                    detection=detections[new_detection_index],
                )

        # Remove dead tracklets.
        logger.info("Removing {} dead tracks.", len(dead_tracklets))
        for track in dead_tracklets:
            self.__active_tracks.remove(track)
            self.__completed_tracks.append(track)

    def __add_new_tracks(
        self, *, assignment_matrix: np.ndarray, detections: np.ndarray
    ) -> None:
        """
        Adds any new tracks to the set of active tracks.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.
            detections: The current detections corresponding to this
                assignment matrix.

        """
        for detection_index, detection in enumerate(detections):
            detection_col = assignment_matrix[:, detection_index]
            if not np.any(detection_col):
                # There is no associated tracklet with this detection,
                # so it represents a new track.
                track = Track()
                logger.info("Adding new track from detection {}.", detection)
                track.add_new_detection(
                    frame_num=self.__frame_num, detection=detection
                )

                self.__active_tracks.add(track)

    def __update_tracks(
        self, *, sinkhorn_matrix: np.ndarray, detections: np.ndarray
    ) -> None:
        """
        Updates the current set of tracks based on the latest tracking result.

        Args:
            sinkhorn_matrix: The assignment matrix between the detections from
                the previous frame and the current one. Should have a shape of
                `[num_detections * num_tracklets]`.
            detections: The current detection bounding boxes. Should have
                shape `[num_detections, 4]`.

        """
        # Un-flatten the assignment matrix.
        num_tracklets = len(self.__previous_geometry)
        num_detections = len(detections)
        sinkhorn_matrix = np.reshape(
            sinkhorn_matrix, (num_tracklets + 1, num_detections + 1)
        )
        logger.debug(sinkhorn_matrix)

        assignment = do_hard_assignment(sinkhorn_matrix).numpy()
        logger.debug(assignment)

        logger.debug("Expanding assignment matrix to {}.", assignment.shape)

        self.__update_active_tracks(
            assignment_matrix=assignment, detections=detections
        )
        self.__add_new_tracks(
            assignment_matrix=assignment, detections=detections
        )

    def __update_saved_state(
        self, *, frame: np.ndarray, geometry: np.ndarray
    ) -> None:
        """
        Updates the saved frames and detections that will be used as the
        input tracks for the next frame.

        Args:
            frame: The current frame image. Should be an array of shape
                `[height, width, channels]`.
            geometry: The geometry for the detections. Should be an array
                of shape `[num_detections, 4]`.

        """
        active_geometry = []
        self.__tracks_by_tracklet_index.clear()

        for i, track in enumerate(self.__active_tracks):
            active_geometry.append(track.last_detection)

            # Save the track object corresponding to this tracklet.
            self.__tracks_by_tracklet_index[i] = track

        self.__previous_frame = frame
        self.__previous_geometry = np.empty((0, 4))
        if len(active_geometry) > 0:
            self.__previous_geometry = np.stack(active_geometry, axis=0)

    def __create_model_inputs(
        self, *, frame: np.ndarray
    ) -> Dict[str, Union[tf.RaggedTensor, tf.Tensor]]:
        """
        Creates an input dictionary for the model based on detections for
        a single frame.

        Args:
            frame: The current frame image. Should be an array of shape
                `[width, height, num_channels]`.

        """
        # Expand dimensions since the model expects a batch.
        frame = np.expand_dims(frame, axis=0)
        previous_frame = np.expand_dims(self.__previous_frame, axis=0)
        previous_geometry = np.expand_dims(self.__previous_geometry, axis=0)

        # Convert to ragged tensors.
        previous_geometry = tf.RaggedTensor.from_tensor(previous_geometry)

        return {
            ModelInputs.DETECTIONS_FRAME.value: frame,
            ModelInputs.TRACKLETS_FRAME.value: previous_frame,
            ModelInputs.TRACKLET_GEOMETRY.value: previous_geometry,
        }

    @staticmethod
    def __add_detection_input(
        inputs: Dict[str, Union[tf.RaggedTensor, tf.Tensor]],
        *,
        detections: np.ndarray,
    ) -> None:
        """
        Adds an input for the current detections to the model inputs.

        Args:
            inputs: The dictionary of model inputs.
            detections: The detections to add.

        """
        # Expand dimensions since the model expects a batch.
        detections = np.expand_dims(detections, axis=0)
        # Convert to ragged tensors.
        detections = tf.RaggedTensor.from_tensor(detections)

        inputs[ModelInputs.DETECTION_GEOMETRY.value] = detections

    def __match_frame_pair(
        self,
        *,
        frame: np.ndarray,
        detection_geometry: Optional[np.ndarray] = None,
    ) -> None:
        """
        Computes the assignment matrix between the current state and new
        detections, and updates the state.

        Args:
            frame: The current frame. Should be an array of shape
                `[height, width, channels]`.
            detection_geometry: Optional detections to use, of shape
                `[num_boxes, 4]`. Otherwise, the detection model will be used.

        """
        model_inputs = self.__create_model_inputs(frame=frame)
        if detection_geometry is None:
            # Apply the detector first.
            logger.info("Applying detection model...")
            model_outputs = self.__detection_model(
                model_inputs, training=False
            )
            detection_geometry = model_outputs[0].numpy()

        num_tracklets = self.__previous_geometry.shape[0]
        num_detections = detection_geometry.shape[0]
        if num_tracklets == 0 or num_detections == 0:
            # Don't bother running the tracker.
            logger.debug("No tracks or no detections, not running tracker.")
            sinkhorn = np.ones(
                (num_tracklets + 1, num_detections + 1), dtype=float
            )
        else:
            logger.info("Applying tracking model...")
            self.__add_detection_input(
                model_inputs, detections=detection_geometry
            )
            model_outputs = self.__tracking_model(model_inputs, training=False)
            sinkhorn = model_outputs[0][0].numpy()

        logger.debug("Got {} detections.", len(detection_geometry))
        # Remove the confidence, since we don't use that for tracking.
        detection_geometry = detection_geometry[:, :4]

        # Update the tracks.
        self.__update_tracks(
            sinkhorn_matrix=sinkhorn, detections=detection_geometry
        )
        # Update the state.
        self.__update_saved_state(frame=frame, geometry=detection_geometry)

    def process_frame(
        self, frame: np.ndarray, detections: Optional[np.ndarray] = None
    ) -> None:
        """
        Use the tracker to process a new frame. It will detect objects in the
        new frame and update the current tracks.

        Args:
            frame: The original image frame from the video. Should be an
                array of shape `[height, width, channels]`.
            detections: Optionally, provide detections instead of using the
                detector model. Should have shape `[num_boxes, 4]`.

        """
        if not self.__maybe_init_state(frame=frame):
            self.__match_frame_pair(frame=frame, detection_geometry=detections)

        self.__frame_num += 1

    @property
    def tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that we have so far.

        """
        return self.__completed_tracks + list(self.__active_tracks)
