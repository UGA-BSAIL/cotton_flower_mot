"""
Implements a KF-based motion model to use for tracking.
"""


from typing import Tuple

import numpy as np
from loguru import logger

from kalmankit import KalmanFilter


class MotionModel:
    """
    Implements a KF-based motion model to use for tracking.
    """

    def __init__(
        self,
        *,
        initial_box: np.array,
        initial_velocity: np.array,
        initial_time: float,
        initial_cov: np.array = np.eye(4),
        process_noise_cov: np.array = np.diag([0.01, 0.01, 0.05, 0.05]),
        observation_noise_cov: np.array = np.diag([0.03, 0.03]),
    ):
        """
        Args:
            initial_box: The initial bounding box of the object, in the form
                `[x, y, w, h]`.
            initial_velocity: The initial state of the object, in the form
                `[vx, vy]`.
            initial_time: The time at which the initial observation was made.
            initial_cov: The initial state covariance estimate. Should be
                a 4x4 matrix.
            process_noise_cov: The process noise covariance matrix. This is
                mostly governed by the fact that the constant velocity
                assumption may not always be correct.
            observation_noise_cov: The observation noise covariance matrix.
                This is mostly impacted by detector inaccuracies.

        """
        state = np.concatenate(
            (
                initial_box[:2].astype(np.float32),
                initial_velocity.astype(np.float32),
            )
        )
        covariance = initial_cov.astype(np.float32).copy()

        # Keeps track of the last time we got an observation.
        self.__observation_time = initial_time
        # Keeps track of the last offset we used for the position state.
        self.__position_offset = self.__compute_center_offset(
            state, box=initial_box
        )
        # Apply the offset to the state.
        state[:2] += self.__position_offset

        # The transition model is very simple in this case since we're
        # assuming constant velocity.
        transition = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        # We can observe the position directly.
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        process_noise_cov = process_noise_cov.astype(np.float32).copy()
        observation_noise_cov = observation_noise_cov.astype(np.float32).copy()

        self.__filter = KalmanFilter(
            A=transition,
            xk=state,
            B=None,
            Pk=covariance,
            H=observation,
            Q=process_noise_cov,
            R=observation_noise_cov,
        )

    @staticmethod
    def __compute_center_offset(state: np.array, *, box: np.array) -> np.array:
        """
        Computes the offset that we will apply to the center point for
        determining the position that we store in the state. This is
        calculated based on the velocity to be the point farthest away from
        the velocity vector.

        Args:
            state: The current state.
            box: The box observation (`[x, y, w, h]`).

        Returns:
            The offset to apply the box center in the state.

        """
        if np.any(state[2:] == 0):
            # If the velocity is zero, we can't compute an offset.
            return np.zeros(2, dtype=np.float32)

        center_x, center_y, width, height = box
        # Find the bottom and left sides of the box.
        box_bottom_y = center_y + height / 2
        box_left_x = center_x - width / 2

        # Find the points where the velocity vector intersects with these sides.
        _, _, vel_x, vel_y = state
        vel_slope = vel_y / vel_x
        bottom_intersection_x = (
            center_x - (box_bottom_y - center_y) / vel_slope
        )
        left_intersection_y = center_y + (center_x - box_left_x) * vel_slope

        if bottom_intersection_x < box_left_x:
            # If this point is outside of the box, it means that the
            # velocity vector intersects on the left side.
            track_point = np.array([box_left_x, left_intersection_y])
        else:
            # Otherwise, it intersects on the bottom.
            track_point = np.array([bottom_intersection_x, box_bottom_y])

        # Compute the offset to the center.
        offset = track_point - box[:2]
        logger.debug("Applying position offset: {}", offset)
        return offset

    def __predict(self, predict_time: float) -> Tuple[np.array, np.array]:
        """
        Predicts the state at some point in the future.

        Args:
            predict_time: The time at which to predict the state.

        Returns:
            The predicted state, as `[x, y, vx, vy]`, and state covariance.
            Note that the state and velocity prediction will be referenced to
            this particular timestep instead of standard units.

        """
        elapsed = predict_time - self.__observation_time
        # Adjust our velocities based on the time that elapsed since the last
        # update.
        xk_adjusted = self.__filter.xk * np.array([1, 1, elapsed, elapsed])
        # Covariances also have to be scaled based on time. Note that we
        # implicitly assume that state covariance remains constant during the
        # time between observations.
        pk_adjusted = self.__filter.Pk * np.sqrt(elapsed)
        q_adjusted = self.__filter.Q * np.sqrt(elapsed)

        return self.__filter.predict(
            Ak=self.__filter.A,
            xk=xk_adjusted,
            Bk=self.__filter.B,
            uk=None,
            Pk=pk_adjusted,
            Qk=q_adjusted,
        )

    def predict(self, predict_time: float) -> Tuple[np.array, np.array]:
        """
        Predicts the state at some point in the future.

        Args:
            predict_time: The time at which to predict the state.

        Returns:
            The predicted state, as `[x, y, vx, vy]`, and state covariance.

        """
        xk_adjusted, pk_adjusted = self.__predict(predict_time)

        # Convert back to standard velocity units.
        elapsed = predict_time - self.__observation_time
        xk_adjusted = xk_adjusted / np.array([1, 1, elapsed, elapsed])

        # Remove the position offset.
        xk_adjusted[:2] -= self.__position_offset
        return xk_adjusted, pk_adjusted

    def add_observation(
        self, observation: np.array, *, observed_time: float
    ) -> None:
        """
        Adds an observation to the model.

        Args:
            observation: The observed bounding box. The observation has the
                form [x, y, w, h].
            observed_time: The time at which the observation was made.

        """
        elapsed = observed_time - self.__observation_time
        if elapsed < 0:
            # This observation is in the past. Don't update.
            logger.warning("Trying to update KF with observation in the past.")
            return

        # Covariances need to be scaled based on time.
        r_adjusted = self.__filter.R * np.sqrt(elapsed)
        # Apply the current offset to the position observation.
        position_observation = observation[:2] + self.__position_offset

        xk_prior, pk_prior = self.__predict(observed_time)
        xk_post, pk_post = self.__filter.update(
            Hk=self.__filter.H,
            xk=xk_prior,
            Pk=pk_prior,
            zk=position_observation,
            Rk=r_adjusted,
        )

        # Now we have to convert back to standard velocity units...
        self.__filter.xk = xk_post / np.array([1, 1, elapsed, elapsed])
        self.__filter.Pk = pk_post / np.sqrt(elapsed)
        self.__observation_time = observed_time

        # Update the offset.
        new_position_offset = self.__compute_center_offset(
            self.__filter.xk, box=observation
        )
        self.__filter.xk[:2] += (new_position_offset - self.__position_offset)
        self.__position_offset = new_position_offset

    @property
    def state(self) -> np.array:
        """
        Returns:
            The current state.

        """
        return self.__filter.xk.copy()

    @property
    def cov(self) -> np.array:
        """
        Returns:
            The current state covariance.

        """
        return self.__filter.Pk.copy()

    @property
    def anchor_point(self) -> np.array:
        """
        Returns:
            The current position of the anchor point (the point
            on the box that is being tracked).

        """
        return self.__filter.xk[:2].copy()
