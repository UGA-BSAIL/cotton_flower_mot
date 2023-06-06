"""
Collection of miscellaneous utility functions used for training by various
pipelines.
"""


from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from .callbacks import ClearMemory
from .schedules import Warmup


def make_learning_rate(
    config: Dict[str, Any]
) -> Union[float, schedules.LearningRateSchedule]:
    """
    Creates the learning rate to use for optimization, based on the user
    configuration.

    Args:
        config: The configuration for the learning rate.

    Returns:
        Either a float for a fixed learning rate, or a `LearningRateSchedule`.

    """
    initial_rate = config["initial"]
    if config.get("decay", False):
        logger.debug("Using decaying learning rate.")
        return tf.keras.experimental.CosineDecayRestarts(
            initial_rate,
            config["decay_steps"],
            t_mul=config["t_mul"],
            m_mul=config["m_mul"],
            alpha=config["min_learning_rate"],
        )

    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps != 0:
        logger.debug("Using LR warmup.")
        return Warmup(max_learning_rate=initial_rate, num_steps=warmup_steps)

    # No decay is configured.
    logger.debug("Using fixed learning rate of {}.", initial_rate)
    return initial_rate


def set_check_numerics(enable: bool) -> None:
    """
    Sets whether to enable checking for NaNs and infinities.

    Args:
        enable: If true, will enable the checks.

    """
    if enable:
        logger.info("Enabling numeric checks. Training might be slow.")
        tf.debugging.enable_check_numerics()
    else:
        tf.debugging.disable_check_numerics()


def bound_numerics(features: tf.Tensor) -> tf.Tensor:
    """
    Enforces numeric limits, by replacing NaNs with zeros, and bounding
    infinite values to near the maximum and minimum float values.

    Args:
        features: The features to check.

    Returns:
        The bounded features.

    """

    def _do_bound_numerics(_features: tf.Tensor) -> tf.Tensor:
        numeric_problems = tf.reduce_any(
            tf.logical_or(tf.math.is_nan(_features), tf.math.is_inf(_features))
        )
        checked_numerics = tf.cond(
            numeric_problems,
            lambda: tf.print(
                f"Tensor {_features.name} has NaN or infinite values:",
                _features,
            ),
            lambda: tf.constant(True),
        )

        with tf.control_dependencies([checked_numerics]):
            dtype = _features.dtype
            _features = tf.clip_by_value(
                _features, dtype.min * 0.9, dtype.max * 0.9
            )
            _features = tf.where(
                tf.math.is_nan(_features), tf.zeros_like(_features), _features
            )

        return _features

    if tf.keras.backend.is_keras_tensor(features):
        # Wrap it in a layer.
        return tf.keras.layers.Lambda(_do_bound_numerics)(features)
    else:
        return _do_bound_numerics(features)


def set_mixed_precision(enable: bool) -> None:
    """
    Sets whether to enable mixed-precision training.

    Args:
        enable: If true, will enable mixed precision.

    """
    if enable:
        logger.info("Enabling mixed-precision training.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        logger.warning("Disabling mixed precision; memory use might be high.")
        tf.keras.mixed_precision.set_global_policy("float32")


@lru_cache
def get_log_dir(tensorboard_output_dir: str) -> Path:
    """
    Gets the logging directory to use for this run.

    Args:
        tensorboard_output_dir: The base tensorboard output directory.

    Returns:
        The run-specific logging directory.

    """
    log_dir = Path(tensorboard_output_dir) / datetime.now().isoformat()
    logger.debug("Writing Tensorboard logs to {}.", log_dir)

    return log_dir


def make_common_callbacks(
    *,
    tensorboard_output_dir: str,
    histogram_period: int,
    update_period: int,
    training_params: Dict[str, Any],
    lr_monitor: str = "val_loss",
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        histogram_period: Period at which to generate histograms for
            Tensorboard output, in epochs.
        update_period: Period in batches at which to log metrics.
        training_params: The hyperparameter configurations for training.
        lr_monitor: Metric to be monitored for automatic LR reduction.

    Returns:
        The list of callbacks.

    """
    # Create a callback for storing Tensorboard logs.
    log_dir = get_log_dir(tensorboard_output_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_period,
        update_freq=update_period,
    )

    nan_termination = tf.keras.callbacks.TerminateOnNaN()

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=lr_monitor,
        patience=training_params["learning_rate"]["lr_patience_epochs"],
        verbose=1,
        min_lr=training_params["learning_rate"]["min_learning_rate"],
    )

    memory_callback = ClearMemory()

    return [
        tensorboard_callback,
        nan_termination,
        reduce_lr_callback,
        memory_callback,
    ]
