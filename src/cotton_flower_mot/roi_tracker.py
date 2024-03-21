"""
Uses the ROI tracking algorithm to speed up the detection process in cases
where there are few flowers in the frame.
"""


from typing import Any, Union, Tuple, List

import numpy as np
import tensorflow as tf

from loguru import logger

from .online_tracker import OnlineTracker, adapt_detection_model, Track
from .tfrt_utils import GraphFunc


class RoiTracker(OnlineTracker):
    """
    Specialized online tracker that uses the ROI tracking algorithm for
    additional performance. This algorithm only applies the detector in small
    regions around known objects when it can get away with it.
    """

    _MAX_ROIS = 1
    """
    This is the maximum number of objects that we will attempt to use ROI 
    detection for. If more objects than this were visible in the previous 
    frame, we should not use ROI detection, because it will actually be 
    slower than running detection on the full frame.
    """

    def __init__(
        self,
        *args: Any,
        roi_detection_model: Union[GraphFunc, tf.keras.Model],
        keyframe_period: float = 0.1,
        roi_size: Tuple[int, int] = (256, 256),
        **kwargs,
    ):
        """

        Args:
            *args: Will be forwarded to the superclass.
            roi_detection_model: The special detection model that operates on
                small input images.
            keyframe_period: The period, in seconds, to skip ROI detection
                and apply the detector to the full image.
            roi_size: The size of the ROI, in pixels (w, h).

        """
        super().__init__(*args, **kwargs)

        logger.info(
            f"ROI tracking configuration:\n"
            f"\tkeyframe_period: {keyframe_period}s,"
            f"\troi_size: {roi_size}"
        )

        self.__roi_detection_model = adapt_detection_model(
            roi_detection_model, reverse_inputs=True
        )
        self.__keyframe_period = keyframe_period
        self.__roi_size = np.array(roi_size[::-1])

        # Timestamp of the last frame we ran the full detector on.
        self.__last_keyframe_time = -np.inf

    def __extract_rois(
        self, frame: np.array, *, tracks: List[Track]
    ) -> List[Tuple[np.array, np.array]]:
        """
        Extracts the ROIs from the frame for the most recent detections of a
        set of tracks.

        Args:
            tracks: The tracks to extract ROIs for.

        Returns:
            The corresponding ROIs that it extracted, and the offsets (in
            pixels) between these ROIs and the input frame.

        """
        logger.debug("Extracting ROIs for {} tracks.", len(tracks))

        rois = []
        for track in tracks:
            center = track.last_detection[:2][::-1]
            # Convert to pixels.
            frame_size = frame.shape[:2]
            center *= frame_size
            center = center.astype(int)

            roi_size_half = self.__roi_size // 2
            top_left = center - roi_size_half
            bottom_right = center + roi_size_half
            np.clip(top_left, 0, frame_size, out=top_left)
            np.clip(bottom_right, 0, frame_size, out=bottom_right)

            y1, x1 = top_left
            y2, x2 = bottom_right
            rois.append((frame[y1:y2, x1:x2], top_left))

        return rois

    @staticmethod
    def __roi_to_frame_detection(
        roi_detection: np.array,
        *,
        roi: np.array,
        frame_offset: np.array,
        frame_size: np.array,
    ) -> np.array:
        """
        Converts a detection from an ROI into (normalized) coordinates of the
        input frame.

        Args:
            roi_detection: The detection in the ROI. Modified in place.
            roi: The original ROI input.
            frame_offset: The offset of the ROI in the full frame, in pixels.
            frame_size: The size of the frame, in the form [h, w].

        """
        norm_offset = frame_offset / frame_size
        norm_offset = norm_offset[::-1]
        roi_size = np.array(roi.shape[:2])
        scale_factor = roi_size / frame_size
        scale_factor = np.tile(scale_factor[::-1], 2)

        roi_detection[:, :4] *= scale_factor
        roi_detection[:, :2] += norm_offset

    def _do_detection(
        self, frame: np.array, *, _frame_time: float
    ) -> Tuple[np.array, np.array]:
        if _frame_time - self.__last_keyframe_time > self.__keyframe_period:
            # It is time to run the full detection.
            logger.debug("Keyframe: running detection on full image.")
            self.__last_keyframe_time = _frame_time
            return super()._do_detection(frame, _frame_time=_frame_time)

        # Find objects visible in the previous frame.
        visible_tracks = []
        for track in self._active_tracks:
            if track.last_detection_frame == track.last_tracked_frame:
                visible_tracks.append(track)
        if len(visible_tracks) > self._MAX_ROIS:
            # Too many objects are visible. Run the full detection.
            logger.debug(
                "Too many objects visible, running detection on full image."
            )
            self.__last_keyframe_time = _frame_time
            return super()._do_detection(frame, _frame_time=_frame_time)

        # Extract ROIs.
        with self._profiler.profile("roi_extraction"):
            rois = self.__extract_rois(frame, tracks=visible_tracks)

        # Detect on ROIs.
        all_boxes = []
        all_appearance = []
        frame_size = frame.shape[:2]
        for roi, frame_offset in rois:
            detector_inputs = self._create_detection_inputs(roi)
            with self._profiler.profile(
                "detection_model_roi", warmup_iters=10
            ):
                detections = self.__roi_detection_model(detector_inputs)

            roi_geometry = detections["geometry"][0].numpy()
            roi_appearance = detections["appearance"][0].numpy()
            if len(roi_geometry) > 0:
                # Convert it back to the coordinates of the full frame.
                self.__roi_to_frame_detection(
                    roi_geometry,
                    roi=roi,
                    frame_offset=frame_offset,
                    frame_size=frame_size,
                )
                all_boxes.append(roi_geometry)
                all_appearance.append(roi_appearance)

        all_boxes_arr = np.empty((0, 5))
        all_appearance_arr = np.empty((0, 1))
        if len(all_boxes) > 0:
            all_boxes_arr = np.concatenate(all_boxes, axis=0)
            all_appearance_arr = np.concatenate(all_appearance, axis=0)
        return all_boxes_arr, all_appearance_arr
