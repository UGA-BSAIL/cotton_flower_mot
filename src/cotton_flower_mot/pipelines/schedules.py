"""
Custom learning rate schedules.
"""

from typing import Any, Dict, Optional

import tensorflow as tf
from loguru import logger


class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements a learning rate warmup that slowly ramps up the learning rate.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        max_learning_rate: float,
        initial_learning_rate: Optional[float] = None
    ):
        """
        Args:
            num_steps: Number of steps to ramp up over.
            max_learning_rate: The maximum learning rate to eventually hit.
            initial_learning_rate: The learning rate to start with. Defaults to
                one hundredth of the maximum.

        """
        self.__num_steps = num_steps
        self.__max_learning_rate = max_learning_rate
        self.__initial_learning_rate = initial_learning_rate

        if self.__initial_learning_rate is None:
            self.__initial_learning_rate = max_learning_rate / 100

        self.__lr_delta = (
            self.__max_learning_rate - self.__initial_learning_rate
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        # Ramp linearly up to the maximum value.
        return tf.cond(
            tf.greater_equal(step, self.__num_steps),
            lambda: self.__max_learning_rate,
            lambda: tf.constant(self.__initial_learning_rate)
            + (
                tf.cast(step, tf.float32)
                / tf.constant(self.__num_steps, dtype=tf.float32)
            )
            * self.__lr_delta,
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            num_steps=self.__num_steps,
            max_learning_rate=self.__max_learning_rate,
            initial_learning_rate=self.__initial_learning_rate,
        )
