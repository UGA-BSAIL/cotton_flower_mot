"""
Framework for online tracking.
"""


from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from ..schemas import ModelInputs, ModelTargets


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
        # Maps frame numbers to detection indices.
        self.__frames_to_indices = {f: i for f, i in enumerate(indices)}
        # Keeps track of the last frame we have a detection for.
        self.__latest_frame = -1

        self.__id = Track._NEXT_ID
        Track._NEXT_ID += 1

    def add_new_detection(
        self, *, frame_num: int, detection_index: int
    ) -> None:
        """
        Adds a new detection to the end of the track.

        Args:
            frame_num: The frame number that this detection is for.
            detection_index: The index of the new detection to add.

        """
        self.__frames_to_indices[frame_num] = detection_index
        self.__latest_frame = max(self.__latest_frame, frame_num)

    @property
    def last_detection_index(self) -> Optional[int]:
        """
        Returns:
            The index of the current last detection in this track, or None
            if the track is empty.

        """
        return self.__frames_to_indices.get(self.__latest_frame)

    @property
    def last_detection_frame(self) -> Optional[int]:
        """
        Returns:
            The frame number at which this object was last detected.

        """
        return self.__latest_frame

    def index_for_frame(self, frame_num: int) -> Optional[int]:
        """
        Gets the corresponding detection index for a particular frame,
        or None if we don't have a detection for that frame.

        Args:
            frame_num: The frame number.

        Returns:
            The index for that frame, or None if we don't have one.

        """
        return self.__frames_to_indices.get(frame_num)

    @property
    def id(self) -> int:
        """
        Returns:
            The unique ID associated with this track.

        """
        return self.__id


class OnlineTracker:
    """
    Performs online tracking using a given model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        *,
        death_window: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model: The model to use for tracking.
            death_window: How many consecutive frames we have to not observe
                a tracklet for before we consider it dead.
            confidence_threshold: The confidence threshold to use for detection.

        """
        self.__model = model
        self.__death_window = death_window
        self.__confidence_threshold = confidence_threshold

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

    def __update_active_tracks(self, assignment_matrix: np.ndarray) -> None:
        """
        Updates the currently-active tracks with new detection information.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.

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
                    detection_index=new_detection_index,
                )

        # Remove dead tracklets.
        logger.info("Removing {} dead tracks.", len(dead_tracklets))
        for track in dead_tracklets:
            self.__active_tracks.remove(track)
            self.__completed_tracks.append(track)

    def __add_new_tracks(self, assignment_matrix: np.ndarray) -> None:
        """
        Adds any new tracks to the set of active tracks.

        Args:
            assignment_matrix: The assignment matrix for the current frame.
                Should have shape `[num_tracklets, num_detections]`.

        """
        num_detections = assignment_matrix.shape[1]

        for detection_index in range(num_detections):
            detection_col = assignment_matrix[:, detection_index]
            if not np.any(detection_col):
                # There is no associated tracklet with this detection,
                # so it represents a new track.
                track = Track()
                logger.info(
                    "Adding new track from detection {}.", detection_index
                )
                track.add_new_detection(
                    frame_num=self.__frame_num, detection_index=detection_index
                )

                self.__active_tracks.add(track)

    def __update_tracks(self, assignment_matrix: np.ndarray) -> None:
        """
        Updates the current set of tracks based on the latest tracking result.

        Args:
            assignment_matrix: The assignment matrix between the detections from
                the previous frame and the current one. Should have a shape of
                `[num_detections * num_tracklets]`.

        """
        # Un-flatten the assignment matrix.
        num_tracklets = len(self.__previous_geometry)
        assignment_matrix = np.reshape(assignment_matrix, (num_tracklets, -1))
        logger.debug(
            "Expanding assignment matrix to {}.", assignment_matrix.shape
        )

        self.__update_active_tracks(assignment_matrix)
        self.__add_new_tracks(assignment_matrix)

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
            if track.last_detection_frame == self.__frame_num:
                # One of our new detections got added to this track. Update
                # it in the state.
                active_geometry.append(geometry[track.last_detection_index])
            else:
                # The track wasn't modified this iteration. Keep the old state.
                active_geometry.append(
                    self.__previous_geometry[track.last_detection_index]
                )

            # Save the track object corresponding to this tracklet.
            self.__tracks_by_tracklet_index[i] = track

        self.__previous_frame = frame
        self.__previous_geometry = np.empty((0,))
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
        # The detections geometry input is not actually used in inference mode,
        # so we just feed it an empty tensor.
        current_geometry = np.empty((1, 0, 4), dtype=np.float32)

        # Convert to ragged tensors.
        previous_geometry = tf.RaggedTensor.from_tensor(previous_geometry)
        current_geometry = tf.RaggedTensor.from_tensor(current_geometry)

        return {
            ModelInputs.DETECTIONS_FRAME.value: frame,
            ModelInputs.TRACKLETS_FRAME.value: previous_frame,
            ModelInputs.TRACKLET_GEOMETRY.value: previous_geometry,
            ModelInputs.DETECTION_GEOMETRY.value: current_geometry,
            # Put the model into inference mode.
            ModelInputs.USE_GT_DETECTIONS.value: np.array([False]),
            # Set the confidence threshold.
            ModelInputs.CONFIDENCE_THRESHOLD.value: np.array(
                [self.__confidence_threshold]
            ),
        }

    def __match_frame_pair(self, *, frame: np.ndarray) -> None:
        """
        Computes the assignment matrix between the current state and new
        detections, and updates the state.

        Args:
            frame: The current frame. Should be an array of shape
                `[height, width, channels]`.

        """
        # Apply the model.
        logger.info("Applying model...")
        model_inputs = self.__create_model_inputs(frame=frame)
        model_outputs = self.__model(model_inputs, training=False)
        print(model_outputs[2])
        print(model_outputs[4])
        assignment = model_outputs[4][0].numpy()
        detection_geometry = model_outputs[2][0].numpy()

        # Update the tracks.
        self.__update_tracks(assignment)
        # Update the state.
        self.__update_saved_state(frame=frame, geometry=detection_geometry)

    def process_frame(self, *, frame: np.ndarray) -> None:
        """
        Use the tracker to process a new frame. It will detect objects in the
        new frame and update the current tracks.

        Args:
            frame: The original image frame from the video. Should be an
                array of shape `[height, width, channels]`.

        """
        if not self.__maybe_init_state(frame=frame):
            self.__match_frame_pair(frame=frame)

        self.__frame_num += 1

    @property
    def tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that we have so far.

        """
        return self.__completed_tracks + list(self.__active_tracks)
