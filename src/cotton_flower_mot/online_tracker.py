"""
Framework for online tracking.
"""


from functools import singledispatch
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.linear_model import RANSACRegressor

from src.cotton_flower_mot.schemas import ModelInputs, ModelTargets
from src.cotton_flower_mot.assignment import (
    do_hard_assignment,
)
from src.cotton_flower_mot.graph_utils import compute_pairwise_similarities
from src.cotton_flower_mot.similarity_utils import compute_ious


GraphFunc = Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
"""
Type alias for a function that runs the TF graph.
"""


class Track:
    """
    Represents a single track.
    """

    _NEXT_ID = 1
    """
    Allows us to associate a unique ID with each track.
    """

    def __init__(self, motion_model_min_detections: int = 3):
        """
        Args:
            motion_model_min_detections: The minimum number of (real)
                detections we need to have before applying the motion model.

        """
        self.__motion_min_detections = motion_model_min_detections

        # Maps frame numbers to detection bounding boxes.
        self.__frames_to_detections = {}
        # Maps frame numbers to appearance features.
        self.__frames_to_appearance = {}
        # Maps frame numbers to whether this detection is real or extrapolated.
        self.__frame_has_detection = {}
        # Keeps track of the last frame we have a detection for.
        self.__latest_frame = -1
        # Keeps track of the last frame we have a motion estimation for.
        self.__latest_motion_frame = -1

        self.__id = Track._NEXT_ID
        Track._NEXT_ID += 1

    def add_new_detection(
        self,
        *,
        frame_num: int,
        detection: np.array,
        appearance_feature: Optional[np.array],
        is_extrapolated: bool = False,
    ) -> None:
        """
        Adds a new detection to the end of the track.

        Args:
            frame_num: The frame number that this detection is for.
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

        self.__frames_to_detections[frame_num] = detection.tolist()
        if appearance_feature is not None:
            self.__frames_to_appearance[
                frame_num
            ] = appearance_feature.tolist()
        self.__frame_has_detection[frame_num] = not is_extrapolated

        if not is_extrapolated:
            self.__latest_frame = max(self.__latest_frame, frame_num)
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
            as estimated by the motion model.

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
        return np.array(self.__frames_to_detections[frame_num])

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
        return np.array(self.__frames_to_appearance[frame_num])

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

    def predict_future_box(self, frame_num: int) -> np.array:
        """
        Extrapolates the trajectory of this object to a future time.

        Args:
            frame_num: The frame number that we are predicting the bounding
                box for.

        Returns:
            The extrapolated bounding box, of the form
            `[center_x, center_y, width, height]`.

        """
        # Get the previous positions. (Only use real detections.)
        frames = [f for f, v in self.__frame_has_detection.items() if v]
        if len(frames) < self.__motion_min_detections:
            raise ValueError(
                "Should have at least three detections to extrapolate."
            )

        boxes = [self.__frames_to_detections[f] for f in frames]
        frames = np.array(frames, dtype=float)
        frames = np.expand_dims(frames, axis=1)
        boxes = np.array(boxes)

        # Perform the regression.
        reg = RANSACRegressor().fit(frames, boxes)

        # Predict the new box.
        return reg.predict([[frame_num]])[0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Gets a dictionary representation of the track that can be easily
        serialized.

        Returns:
            A dictionary representing the track.

        """
        return dict(
            motion_model_min_detections=self.__motion_min_detections,
            frames_to_detections=self.__frames_to_detections,
            frame_has_detection=self.__frame_has_detection,
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
            motion_model_min_detections=config["motion_model_min_detections"]
        )

        track.__frames_to_detections = config["frames_to_detections"]
        track.__frame_has_detection = config["frame_has_detection"]
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
        death_window: int = 10,
        motion_model_min_detections: int = 5,
        confidence_threshold: float = 0.0,
        stage_one_iou_threshold: float = 0.5,
    ):
        """
        Args:
            tracking_model: The model to use for tracking.
            detection_model: The model to use for tracking.
            death_window: How many consecutive frames we have to not observe
                a tracklet for before we consider it dead.
            motion_model_min_detections: Minimum number of (real) detections
                we must have before applying the motion model.
            confidence_threshold: The confidence threshold to use for the
                detector.
            stage_one_iou_threshold: The IOU threshold to use for the
                first-stage fast association step.

        """
        self.__tracking_model = _adapt_tracking_model(tracking_model)
        self.__detection_model = _adapt_detection_model(detection_model)
        self.__death_window = death_window
        self.__motion_min_detections = motion_model_min_detections
        self.__confidence_threshold = confidence_threshold
        self.__iou_threshold = tf.constant(stage_one_iou_threshold)

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

    def __update_active_tracks(
        self,
        *,
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
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
                    # Otherwise, extrapolate a new bounding box based on
                    # previous track information.
                    try:
                        extrapolated_box = track.predict_future_box(
                            self.__frame_num
                        )
                    except ValueError:
                        logger.debug(
                            "Not extrapolating track because there "
                            "are too few detections."
                        )
                        continue
                    track.add_new_detection(
                        frame_num=self.__frame_num,
                        detection=extrapolated_box,
                        appearance_feature=None,
                        is_extrapolated=True,
                    )

            else:
                # Find the associated detection.
                new_detection_index = np.argmax(
                    assignment_matrix[tracklet_index]
                )
                track.add_new_detection(
                    frame_num=self.__frame_num,
                    detection=detections[new_detection_index],
                    appearance_feature=appearances[new_detection_index],
                )

        # Remove dead tracklets.
        logger.info("Removing {} dead tracks.", len(dead_tracklets))
        for track in dead_tracklets:
            self.__active_tracks.remove(track)
            self.__completed_tracks.append(track)

    def __add_new_tracks(
        self,
        *,
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
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

        """
        for detection_index, (detection, appearance) in enumerate(
            zip(detections, appearances)
        ):
            detection_col = assignment_matrix[:, detection_index]
            if not np.any(detection_col):
                # There is no associated tracklet with this detection,
                # so it represents a new track.
                track = Track(
                    motion_model_min_detections=self.__motion_min_detections
                )
                logger.info("Adding new track from detection {}.", detection)
                track.add_new_detection(
                    frame_num=self.__frame_num,
                    detection=detection,
                    appearance_feature=appearance,
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
        assignment_matrix: np.array,
        detections: np.array,
        appearances: np.array,
    ) -> None:
        """
        Updates the current set of tracks based on the latest tracking result.

        Args:
            assignment_matrix: The boolean assignment matrix between the
                detections from the previous frame and the current one.
                Should have a shape of `[num_tracklets, num_detections]`.
            detections: The current detection bounding boxes. Should have
                shape `[num_detections, 4]`.
            appearances: The current appearance features. Should have shape
                `[num_detections, num_channels]`.

        """
        logger.debug(assignment_matrix)

        self.__update_active_tracks(
            assignment_matrix=assignment_matrix,
            detections=detections,
            appearances=appearances,
        )
        self.__add_new_tracks(
            assignment_matrix=assignment_matrix,
            detections=detections,
            appearances=appearances,
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

    def __do_fast_association(self, geometry: np.array) -> Optional[np.array]:
        """
        Performs an initial fast attempt at association based on the bounding
        box IOUs.

        Args:
            geometry: The current bounding boxes.

        Returns:
            The boolean assignment matrix of shape `[num_tracklets,
            num_detections], if association succeeded, otherwise `None`.

        """
        # First, compute IOUs between all tracklets and all detections.
        geometry = tf.expand_dims(
            tf.convert_to_tensor(geometry, dtype=tf.float32), axis=0
        )
        previous_geometry = tf.expand_dims(
            tf.convert_to_tensor(self.__previous_geometry, dtype=tf.float32),
            axis=0,
        )
        pairwise_ious = compute_pairwise_similarities(
            compute_ious,
            left_features=previous_geometry,
            right_features=geometry,
        )[0]

        # To meet the criteria for a valid association, each detection must
        # have AT MOST ONE plausible association with a tracklet.
        valid_matches = tf.greater(pairwise_ious, self.__iou_threshold)
        valid_matches_int = tf.cast(valid_matches, tf.int32)
        num_tracklets_matches = tf.reduce_sum(
            valid_matches_int, axis=0
        ).numpy()
        num_detections_matches = tf.reduce_sum(
            valid_matches_int, axis=1
        ).numpy()

        print(valid_matches_int)
        if np.any(num_tracklets_matches > 1) or np.any(
            num_detections_matches > 1
        ):
            # Ambiguous association.
            return None

        # Also, we can't have any tracklet/detection pairs that could have
        # been matched but weren't.
        if np.sum(valid_matches_int) < np.min(valid_matches_int.shape):
            return None

        # Otherwise, this is a valid association.
        return valid_matches.numpy()

    def __do_association(
        self, *, detection_geometry: np.array, appearance_features: np.array
    ) -> None:
        """
        Performs the association step of the tracking pipeline.

        Args:
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
            assignment := self.__do_fast_association(detection_geometry)
        ) is None:
            # Fast association failed.
            logger.info("Falling back on slow association...")
            model_inputs = self.__create_tracking_inputs(
                detections=detection_geometry,
                appearance_features=appearance_features,
            )
            model_outputs = self.__tracking_model(model_inputs)
            sinkhorn = model_outputs[ModelTargets.SINKHORN.value][0].numpy()
            assignment = self.__sinkhorn_to_assigment(
                sinkhorn, detections=detection_geometry
            )

        logger.debug("Got {} detections.", len(detection_geometry))
        # Remove the confidence, since we don't use that for tracking.
        detection_geometry = detection_geometry[:, :4]

        # Update the tracks.
        self.__update_tracks(
            assignment_matrix=assignment,
            detections=detection_geometry,
            appearances=appearance_features,
        )

    def __match_frame_pair(
        self,
        *,
        frame: np.ndarray,
    ) -> None:
        """
        Computes the assignment matrix between the current state and new
        detections, and updates the state.

        Args:
            frame: The current frame. Should be an array of shape
                `[height, width, channels]`.

        """
        model_inputs = self.__create_detection_inputs(frame=frame)
        # Apply the detector first.
        logger.info("Applying detection model...")
        detections = self.__detection_model(model_inputs)
        detection_geometry = detections["geometry"][0].numpy()
        appearance_features = detections["appearance"][0].numpy()
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
        )

        # Update the state.
        self.__update_saved_state(frame=frame)

    def process_frame(
        self,
        frame: np.array,
    ) -> None:
        """
        Use the tracker to process a new frame. It will detect objects in the
        new frame and update the current tracks.

        Args:
            frame: The original image frame from the video. Should be an
                array of shape `[height, width, channels]`.

        """
        if not self.__maybe_init_state(frame=frame):
            self.__match_frame_pair(
                frame=frame,
            )

        self.__frame_num += 1

    @property
    def tracks(self) -> List[Track]:
        """
        Returns:
            All the tracks that we have so far.

        """
        return self.__completed_tracks + list(self.__active_tracks)
