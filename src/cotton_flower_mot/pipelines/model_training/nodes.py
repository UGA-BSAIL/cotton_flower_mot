"""
Nodes for the model training pipeline.
"""


from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from ..config import ModelConfig
from ..schemas import ModelTargets
from .callbacks import LogHeatmaps
from .centernet_model import build_model
from .losses import make_losses
from .metrics import make_metrics
from ..learning_rate import make_learning_rate


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


def create_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    model = build_model(config)
    logger.info("Model has {} parameters.", model.count_params())

    return model


def _make_callbacks(
    *,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tensorboard_output_dir: str,
    histogram_period: int,
    update_period: int,
    heatmap_size: Tuple[int, int],
    heatmap_period: int,
    num_heatmap_batches: int,
    num_heatmap_images: int,
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        model: The model to log heatmaps from.
        dataset: The dataset to use for logging heatmaps.
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        histogram_period: Period at which to generate histograms for
            Tensorboard output, in epochs.
        update_period: Period in batches at which to log metrics.
        heatmap_size: Size of the logged heatmap visualizations.
            (width, height)
        heatmap_period: Period at which to generate heatmap visualizations,
            in epochs.
        num_heatmap_batches: Total number of batches to log heatmap data from.
        num_heatmap_images: Total number of heatmap images to include in each
            batch.

    Returns:
        The list of callbacks.

    """
    # Create a callback for storing Tensorboard logs.
    log_dir = Path(tensorboard_output_dir) / datetime.now().isoformat()
    logger.debug("Writing Tensorboard logs to {}.", log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_period,
        update_freq=update_period,
    )

    nan_termination = tf.keras.callbacks.TerminateOnNaN()

    heatmap_callback = LogHeatmaps(
        model=model,
        dataset=dataset,
        log_dir=log_dir / "heatmaps",
        resize_images=heatmap_size,
        log_period=heatmap_period,
        max_num_batches=num_heatmap_batches,
        num_images_per_batch=num_heatmap_images,
    )

    return [tensorboard_callback, nan_termination, heatmap_callback]


def _remove_unused_targets(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Removes unused targets from a dataset.

    Args:
        dataset: The dataset to remove targets from.

    Returns:
        The modified dataset.

    """

    def _remove_targets(inputs: Dict, targets: Dict) -> Tuple[Dict, Dict]:
        targets.pop(ModelTargets.SINKHORN.value)
        targets.pop(ModelTargets.ASSIGNMENT.value)

        return inputs, targets

    return dataset.map(_remove_targets)


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    validation_frequency: int = 1,
    loss_params: Dict[str, Any],
    heatmap_loss_weight: float = 1.0,
    geometry_loss_weight: float = 1.0,
    ciou_loss_weight: float = 1.0,
    **kwargs: Any,
) -> tf.keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        validation_frequency: Number of training epochs after which to run
            validation.
        loss_params: Parameters to pass to the loss functions.
        heatmap_loss_weight: The loss weight for the heatmap focal loss.
        geometry_loss_weight: The loss weight for the L1 geometry loss.
        ciou_loss_weight: The loss weight for the cIOU loss.
        **kwargs: Will be forwarded to `_make_callbacks()`.

    Returns:
        The trained model.

    """
    # Clear unused tracking targets.
    training_data = _remove_unused_targets(training_data)
    testing_data = _remove_unused_targets(testing_data)

    callbacks = _make_callbacks(model=model, dataset=testing_data, **kwargs)

    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=make_learning_rate(phase["learning_rate"]),
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(**loss_params),
            loss_weights={
                ModelTargets.HEATMAP.value: heatmap_loss_weight,
                ModelTargets.GEOMETRY_DENSE_PRED.value: geometry_loss_weight,
                ModelTargets.GEOMETRY_SPARSE_PRED.value: ciou_loss_weight,
            },
            metrics=make_metrics(),
        )
        model.fit(
            training_data,
            validation_data=testing_data,
            epochs=phase["num_epochs"],
            callbacks=callbacks,
            validation_freq=validation_frequency,
        )

    return model
