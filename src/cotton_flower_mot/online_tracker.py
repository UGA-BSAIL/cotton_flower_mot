import abc
from typing import Dict

import numpy as np


class OnlineTracker(abc.ABC):
    """
    Implements the association between two frames in an online tracking system.
    """

    @abc.abstractmethod
    def associate(self, tracking_inputs: Dict[str, np.array]) -> np.array:
        """
        Associates tracklets with new detections.

        Args:
            tracking_inputs: Contains the inputs for the tracking system.
                This dictionary should contain the following keys:
                - "detections": The detection bounding boxes.
                - "detections_appearance": The detection appearance features.
                - "tracklets": The tracklet bounding boxes.
                - "tracklets_appearance": The tracklet appearance features.

        Returns:
            The sparse assignment matrix, an array with two columns where the
            first column is the index of the tracklet and the second column
            is the index of the corresponding detection in the input.

        """