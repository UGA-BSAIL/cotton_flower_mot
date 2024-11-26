from typing import Optional, Tuple, Dict

import tensorflow as tf

import numpy as np

from loguru import logger

from .assignment import do_hard_assignment
from .graph_utils import compute_pairwise_similarities
from .model import TrackingModel
from .online_tracker import OnlineTracker
from .profiler import ProfilingManager
from .similarity_utils import compute_ious


class ModelTracker(OnlineTracker):
    """
    An online tracker that uses a model to track objects. It has an optimal
    first stage which will attempt to solve easy cases using a faster SORT-like
    algorithm.
    """

    def __init__(
        self,
        tracking_model: TrackingModel,
        stage_one_iou_threshold: float = 0.5,
        enable_two_stage_association: bool = True,
        profile_manager: ProfilingManager = ProfilingManager(),
    ):
        """
        Args:
            tracking_model: The model to use for tracking.
            stage_one_iou_threshold: The IOU threshold to use for the
                first-stage fast association step.
            enable_two_stage_association: If true, it will attempt a fast
                IOU-based association process and only fall back on the GNN
                if that fails.
            profile_manager: The profiling manager to use for profiling.

        """
        self.__tracking_model = tracking_model
        self.__iou_threshold = tf.constant(stage_one_iou_threshold)
        self.__enable_fast_association = enable_two_stage_association
        logger.info(
            f"Model tracker configuration:\n"
            f"\tTwo-stage association: {self.__enable_fast_association}\n"
            f"\tIOU threshold: {self.__iou_threshold}"
        )

        self._profiler = profile_manager

    def __sinkhorn_to_assigment(
        self,
        sinkhorn_matrix: np.array,
        *,
        num_detections: int,
        num_tracklets: int,
    ) -> np.array:
        """
        Converts a sinkhorn matrix to a hard assignment matrix.

        Args:
            sinkhorn_matrix: The sinkhorn matrix between the detections from
                the previous frame and the current one. Should have a shape of
                `[num_detections * num_tracklets]`.
            num_detections: The number of detections.
            num_tracklets:  The number of tracklets.

        Returns:
            The hard assignment matrix.

        """
        # Un-flatten the sinkhorn matrix.
        sinkhorn_matrix = np.reshape(
            sinkhorn_matrix, (num_tracklets + 1, num_detections + 1)
        )
        logger.debug(sinkhorn_matrix)

        assignment = do_hard_assignment(sinkhorn_matrix).numpy()
        logger.debug("Expanding assignment matrix to {}.", assignment.shape)

        return assignment

    @staticmethod
    def __dense_to_sparse_assignment(
        assignment: np.array,
        row_indices: Optional[np.array] = None,
        col_indices: Optional[np.array] = None,
    ) -> np.array:
        """
        Converts a dense assignment matrix to a sparse representation.

        Args:
            assignment: The dense assignment matrix.
            row_indices: Optional row indices to use. If not specified,
                it will just use 0 to the # of rows.
            col_indices: Optional column indices to use. If not specified,
                it will just use 0 to the # of columns.

        Returns:
            A 2D array where the first column is the row indices and the
            second column is the corresponding column indices.

        """
        if row_indices is None:
            row_indices = np.arange(assignment.shape[0])
        if col_indices is None:
            col_indices = np.arange(assignment.shape[1])

        col_indices_tiled = np.tile(col_indices, (len(row_indices), 1))
        corresponding_cols = col_indices_tiled[assignment]

        # Get only the row indices that are actually assigned.
        row_indices_mask = np.any(assignment, axis=1)
        row_indices = row_indices[row_indices_mask]

        return np.stack((row_indices, corresponding_cols), axis=1)

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
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Implementation of the fast association routine.

        Args:
            geometry: The current bounding boxes.
            previous_geometry: The previous bounding boxes.
            iou_threshold: The IOU threshold to use for determining matches.

        Returns:
            The boolean assignment matrix of shape `[num_tracklets,
            num_detections], the indices of the previous bounding boxes that
            were not matched, and the indices of the current bounding boxes
            that were not matched.

        """
        # First, compute IOUs between all tracklets and all detections.
        pairwise_ious = compute_pairwise_similarities(
            compute_ious,
            left_features=previous_geometry[None, :, :],
            right_features=geometry[None, :, :],
        )[0]

        valid_matches = tf.greater(pairwise_ious, iou_threshold)
        valid_matches_int = tf.cast(valid_matches, tf.int32)
        num_detections_matches = tf.reduce_sum(valid_matches_int, axis=0)
        num_tracklets_matches = tf.reduce_sum(valid_matches_int, axis=1)

        # To meet the criteria for a valid association, detection must have
        # AT MOST ONE plausible association with a tracklet.
        unique_detections_matches = tf.less_equal(num_detections_matches, 1)
        unique_tracklets_matches = tf.less_equal(num_tracklets_matches, 1)
        unique_match_mask = tf.logical_and(
            tf.reshape(unique_detections_matches, (1, -1)),
            tf.reshape(unique_tracklets_matches, (-1, 1)),
        )
        valid_matches = tf.where(unique_match_mask, valid_matches, False)

        # We also want to return the inputs that were not matched so we can
        # pass them on to second stage association.
        detections_matched_mask = tf.reduce_any(valid_matches, axis=0)
        tracklets_matched_mask = tf.reduce_any(valid_matches, axis=1)
        detections_unmatched_indices = tf.where(~detections_matched_mask)
        tracklets_unmatched_indices = tf.where(~tracklets_matched_mask)
        detections_unmatched_indices = tf.squeeze(
            detections_unmatched_indices, axis=1
        )
        tracklets_unmatched_indices = tf.squeeze(
            tracklets_unmatched_indices, axis=1
        )

        return (
            valid_matches,
            tracklets_unmatched_indices,
            detections_unmatched_indices,
        )

    def __do_fast_association(
        self, model_inputs: Dict[str, np.array]
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Performs an initial fast attempt at association based on the bounding
        box IOUs.

        Args:
            model_inputs: For simplicity, we take inputs in the same form
                that the tracking model does, i.e. as a dictionary of inputs.

        Returns:
            The sparse boolean assignment matrix, the indices of the
            tracklets that were NOT matched successfully, and the indices of
            the detections that were NOT matched successfully.

        """
        geometry = model_inputs["detections"]
        previous_geometry = model_inputs["tracklets"]

        with self._profiler.profile("fast_association", warmup_iters=10):
            geometry = tf.convert_to_tensor(geometry, dtype=tf.float32)
            previous_geometry = tf.convert_to_tensor(
                previous_geometry, dtype=tf.float32
            )

            assignment, failed_tracklet_ind, failed_detection_ind = (
                o.numpy()
                for o in self._fast_association_impl(
                    geometry, previous_geometry, self.__iou_threshold
                )
            )
            assignment = self.__dense_to_sparse_assignment(assignment)
            if len(failed_detection_ind) + len(failed_tracklet_ind) > 0:
                logger.debug(
                    "Fast association failed to match {} detections and {} "
                    "tracklets.",
                    len(failed_detection_ind),
                    len(failed_tracklet_ind),
                )

            return assignment, failed_tracklet_ind, failed_detection_ind

    def associate(self, tracking_inputs: Dict[str, np.array]) -> np.array:
        num_tracklets = tracking_inputs["tracklets"].shape[0]
        num_detections = tracking_inputs["detections"].shape[0]
        # Detection and track indices to use for slow association, if necessary.
        slow_tracklet_ind = np.arange(num_tracklets)
        slow_detection_ind = np.arange(num_detections)
        # Assignment matrix from fast association.
        fast_assignment = np.empty((0, 2), dtype=int)
        slow_assignment = np.empty_like(fast_assignment)

        if num_tracklets == 0 or num_detections == 0:
            # Don't bother running the tracker.
            logger.debug("No tracks or no detections, not running tracker.")
        elif self.__enable_fast_association:
            # Try to fast-associate as much as we can.
            fast_assignment, slow_tracklet_ind, slow_detection_ind = (
                self.__do_fast_association(tracking_inputs)
            )
        if len(slow_tracklet_ind) > 0 and len(slow_detection_ind) > 0:
            # Fast association failed for some of the inputs.
            logger.debug("Falling back on slow association...")
            with self._profiler.profile("slow_association", warmup_iters=10):
                # Filter to only the inputs we need for slow association.
                slow_inputs = dict(
                    detections=tracking_inputs["detections"][
                        slow_detection_ind
                    ],
                    detections_appearance=tracking_inputs[
                        "detections_appearance"
                    ][slow_detection_ind],
                    tracklets=tracking_inputs["tracklets"][slow_tracklet_ind],
                    tracklets_appearance=tracking_inputs[
                        "tracklets_appearance"
                    ][slow_tracklet_ind],
                )

                sinkhorn = self.__tracking_model.track(**slow_inputs)

                slow_assignment = self.__sinkhorn_to_assigment(
                    sinkhorn,
                    num_detections=len(slow_detection_ind),
                    num_tracklets=len(slow_tracklet_ind),
                )
                slow_assignment = self.__dense_to_sparse_assignment(
                    slow_assignment,
                    row_indices=slow_tracklet_ind,
                    col_indices=slow_detection_ind,
                )

        # Combine results from fast and slow association.
        return np.concatenate((fast_assignment, slow_assignment), axis=0)
