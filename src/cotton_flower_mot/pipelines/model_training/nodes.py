"""
Nodes for the model training pipeline.
"""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger
from keras import layers
from keras.applications.efficientnet_v2 import EfficientNetV2S

from ..callbacks import LogHeatmaps, KeepBest
from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from ..training_utils import (
    get_log_dir,
    make_common_callbacks,
)
from .combined_model import build_combined_model, build_separate_models
from .losses import make_losses
from .metrics import make_metrics


def prepare_pretrained_encoder(
    encoder: tf.keras.Model, config: ModelConfig, freeze_fraction: float = 1.0
) -> tf.keras.Model:
    """
    Prepares a custom pretrained model to be used as an encoder. In this case,
    the model is expected to be an EfficientNetV2 model. It will be frozen and
    have the proper layers extracted.

    Args:
        encoder: The model to use as the encoder.
        config: The model configuration.
        freeze_fraction: Fraction of the layers to freeze. A number <1 will
            allow layers at the top to be trained while keeping the bottom
            frozen.

    Returns:
        A modified model.

    """
    logger.info("Using a custom pre-trained encoder.")

    # Change the input size of the model.
    new_input = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    # It's easiest here to create a fresh encoder and manually copy pre-trained
    # weights.
    new_encoder = EfficientNetV2S(
        include_top=False,
        input_tensor=new_input,
        input_shape=config.detection_model_input_shape,
        weights="imagenet",
    )

    # Calculate how many layers to freeze.
    num_layers = len(new_encoder.layers)
    num_to_freeze = int(num_layers * freeze_fraction)
    logger.debug("Freezing {} layers out of {}.", num_to_freeze, num_layers)

    # Copy over the pre-trained weights.
    for i, new_layer in enumerate(new_encoder.layers[1:]):
        old_layer = encoder.get_layer(new_layer.name)

        if i <= num_to_freeze:
            weights = old_layer.get_weights()
            if new_layer.name == "stem_conv" and weights[0].shape[2] == 1:
                # For a colorization model, expand the weights to support
                # color input.
                logger.debug("Expanding model weights for color input.")
                weights[0] = np.concatenate([weights[0]] * 3, axis=2) / 3.0

            new_layer.set_weights(weights)
            new_layer.trainable = False
        else:
            new_layer.trainable = True

    # Extract the layers that we need.
    block2 = new_encoder.get_layer("block2d_add").get_output_at(0)
    block3 = new_encoder.get_layer("block3d_add").get_output_at(0)
    block5 = new_encoder.get_layer("block5i_add").get_output_at(0)
    top = new_encoder.get_layer("top_activation").get_output_at(0)

    return tf.keras.Model(
        inputs=new_encoder.inputs, outputs=[block2, block3, block5, top]
    )


def create_model(
    config: ModelConfig, encoder: Optional[tf.keras.Model] = None
) -> tf.keras.Model:
    """
    Builds the model to use.

    Args:
        config: The model configuration.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.

    Returns:
        The end-to-end model.

    """
    detector, tracker = build_separate_models(config, encoder=encoder)
    combined = build_combined_model(config, detector=detector, tracker=tracker)
    logger.info("Model has {} parameters.", combined.count_params())

    return combined


def _make_callbacks(
    *,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tensorboard_output_dir: str,
    heatmap_size: Tuple[int, int],
    heatmap_period: int,
    num_heatmap_batches: int,
    num_heatmap_images: int,
    **kwargs: Any,
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        model: The model to log heatmaps from.
        dataset: The dataset to use for logging heatmaps.
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        heatmap_size: Size of the logged heatmap visualizations.
            (width, height)
        heatmap_period: Period at which to generate heatmap visualizations,
            in epochs.
        num_heatmap_batches: Total number of batches to log heatmap data from.
        num_heatmap_images: Total number of heatmap images to include in each
            batch.
        **kwargs: Will be forwarded to `make_common_callbacks`.

    Returns:
        The list of callbacks.

    """
    common_callbacks = make_common_callbacks(
        tensorboard_output_dir=tensorboard_output_dir, **kwargs
    )

    log_dir = get_log_dir(tensorboard_output_dir)
    heatmap_callback = LogHeatmaps(
        model=model,
        dataset=dataset,
        log_dir=log_dir / "heatmaps",
        resize_images=heatmap_size,
        log_period=heatmap_period,
        max_num_batches=num_heatmap_batches,
        num_images_per_batch=num_heatmap_images,
    )

    return common_callbacks + [heatmap_callback]


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
    sinkhorn_loss_weight: float = 1.0,
    **kwargs: Any,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Trains a model.

    Args:
        model: The end-to-end model. This should share weights with the
            other two models.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        validation_frequency: Number of training epochs after which to run
            validation.
        loss_params: Parameters to pass to the loss functions.
        heatmap_loss_weight: The loss weight for the heatmap focal loss.
        geometry_loss_weight: The loss weight for the L1 geometry loss.
        sinkhorn_loss_weight: The loss weight for the sinkhorn loss.
        **kwargs: Will be forwarded to `_make_callbacks()`.

    Returns:
        The trained model from the best epoch and from the last epoch.

    """
    # Add a callback for keeping track of the best model.
    best_model_callback = KeepBest()

    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        callbacks = _make_callbacks(
            model=model,
            dataset=testing_data,
            training_params=phase,
            **kwargs,
        )
        callbacks.append(best_model_callback)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=phase["learning_rate"]["initial"]
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(**loss_params),
            loss_weights={
                ModelTargets.HEATMAP.value: heatmap_loss_weight,
                ModelTargets.GEOMETRY_DENSE_PRED.value: geometry_loss_weight,
                ModelTargets.SINKHORN.value: sinkhorn_loss_weight,
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

    return best_model_callback.best_model, model
