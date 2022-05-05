"""
Nodes for pre-training the RotNet model.
"""


from pathlib import Path
from typing import Any, Dict, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger

from ..dataset_io import (
    RotNetFeatureName,
    rot_net_inputs_and_targets_from_imagenet,
)
from ..learning_rate import make_learning_rate
from ..model_training.layers.resnet import rotnet_resnet
from ..model_training.losses import make_rotnet_loss
from ..model_training.metrics import make_rotnet_metrics


def set_mixed_precision(use_mixed: bool) -> None:
    """
    Set whether to use mixed precision to speed up training.

    Args:
        use_mixed: Whether to use mixed precision.

    """
    if use_mixed:
        logger.info("Mixed precision training is on.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def create_model() -> tf.keras.Model:
    """
    Builds the model to use.

    Returns:
        The model that it created.

    """
    images = tf.keras.Input(
        shape=(224, 224, 3), name=RotNetFeatureName.IMAGE.value
    )

    def _normalize(_images: tf.Tensor) -> tf.Tensor:
        # Normalize the images before putting them through the model.
        float_images = tf.cast(_images, tf.keras.backend.floatx())
        # return tf.image.per_image_standardization(float_images)
        return tf.keras.applications.resnet_v2.preprocess_input(float_images)

    normalized = tf.keras.layers.Lambda(_normalize, name="normalize")(images)

    model = rotnet_resnet(normalized)
    logger.info("Model has {} parameters.", model.count_params())

    return model


def load_datasets(
    dataset_path: Union[Path, str]
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads training and testing datasets for training RotNet.

    Args:
        dataset_path: The path to load datasets from.

    Returns:
        The training and testing datasets.

    """
    logger.info("Loading ImageNet from {}.", dataset_path)
    dataset_path = Path(dataset_path)
    imagenet_splits = tfds.load(
        name="imagenet2012", data_dir=dataset_path.as_posix()
    )

    rot_net_train = rot_net_inputs_and_targets_from_imagenet(
        imagenet_splits["train"]
    )
    rot_net_test = rot_net_inputs_and_targets_from_imagenet(
        imagenet_splits["validation"]
    )

    return rot_net_train, rot_net_test


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    lr_config: Dict[str, Any],
    num_epochs: int,
    validation_frequency: int = 1,
) -> tf.keras.Model:
    """
    Pre-trains a RotNet model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        lr_config: The configuration to use for the learning rate schedule.
        num_epochs: The number of epochs to train for.
        validation_frequency: Number of training epochs after which to run
            validation.

    Returns:
        The trained model.

    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=make_learning_rate(lr_config),
        momentum=lr_config["momentum"],
    )
    model.compile(
        optimizer=optimizer,
        loss=make_rotnet_loss(),
        metrics=make_rotnet_metrics(),
    )
    model.fit(
        training_data,
        validation_data=testing_data,
        epochs=num_epochs,
        validation_freq=validation_frequency,
    )

    return model
