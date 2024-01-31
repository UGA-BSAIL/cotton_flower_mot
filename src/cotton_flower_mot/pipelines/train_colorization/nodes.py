"""
Nodes for the `train_colorization` pipeline.
"""


from typing import Any, Dict, List

import tensorflow as tf
from loguru import logger

from ..config import ModelConfig
from ..model_training.centernet_model import build_colorization_model
from src.cotton_flower_mot.training_utils import make_common_callbacks, make_learning_rate
from src.cotton_flower_mot.schemas import ColorizationTargets


def create_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    model = build_colorization_model(config)
    logger.info("Model has {} parameters.", model.count_params())

    return model


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    **kwargs: Any,
) -> tf.keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        **kwargs: Will be forwarded to `make_common_callbacks()`.

    Returns:
        The trained model.

    """
    # Temporarily restrict the amount of training data for rapid development.
    training_data = training_data.take(20000)

    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        # We don't do any validation here, so monitor training loss.
        callbacks = make_common_callbacks(
            training_params=phase, lr_monitor="loss", **kwargs
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=make_learning_rate(phase["learning_rate"]),
        )
        model.compile(
            optimizer=optimizer,
            loss={
                ColorizationTargets.HUE_HIST.value: "kl_divergence",
                ColorizationTargets.CHROMA_HIST.value: "kl_divergence",
            },
        )
        model.fit(
            training_data,
            epochs=phase["num_epochs"],
            callbacks=callbacks,
        )

    return model
