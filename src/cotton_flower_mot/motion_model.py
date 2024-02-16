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
        initial_state: np.array,
        initial_time: float,
        initial_cov: np.array = np.eye(4),
        process_noise_cov: np.array = np.diag([1, 1, 50, 50]),
        observation_noise_cov: np.array = np.diag([10, 10]),
    ):
        """
        Args:
            initial_state: The initial state of the object. The state has the
                form [x, y, vx, vy].
            initial_time: The time at which the initial observation was made.
            initial_cov: The initial state covariance estimate. Should be
                a 4x4 matrix.
            process_noise_cov: The process noise covariance matrix. This is
                mostly governed by the fact that the constant velocity
                assumption may not always be correct.
            observation_noise_cov: The observation noise covariance matrix.
                This is mostly impacted by detector inaccuracies.

        """
        state = initial_state.astype(np.float32).copy()
        covariance = initial_cov.astype(np.float32).copy()
        self.__observation_time = initial_time

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
        return (
            xk_adjusted / np.array([1, 1, elapsed, elapsed]),
            pk_adjusted / np.sqrt(elapsed),
        )

    def add_observation(
        self, observation: np.array, *, observed_time: float
    ) -> None:
        """
        Adds an observation to the model.

        Args:
            observation: The observation to add. The observation has the
                form [x, y].
            observed_time: The time at which the observation was made.

        """
        elapsed = observed_time - self.__observation_time
        if elapsed < 0:
            # This observation is in the past. Don't update.
            logger.warning("Trying to update KF with observation in the past.")
            return

        # Covariances need to be scaled based on time.
        r_adjusted = self.__filter.R * np.sqrt(elapsed)

        xk_prior, pk_prior = self.__predict(observed_time)
        xk_post, pk_post = self.__filter.update(
            Hk=self.__filter.H,
            xk=xk_prior,
            Pk=pk_prior,
            zk=observation,
            Rk=r_adjusted,
        )

        # Now we have to convert back to standard velocity units...
        self.__filter.xk = xk_post / np.array([1, 1, elapsed, elapsed])
        self.__filter.Pk = pk_post / np.sqrt(elapsed)
        self.__observation_time = observed_time

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
