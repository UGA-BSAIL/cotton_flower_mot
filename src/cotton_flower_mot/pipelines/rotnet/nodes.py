"""
Nodes for pre-training the RotNet model.
"""


import tensorflow as tf
from loguru import logger
from ..model_training.layers.resnet import rotnet_resnet
from ..model_training.losses import make_rotnet_loss
from ..learning_rate import make_learning_rate
from typing import Dict, Any, Union, Tuple
from pathlib import Path
import tensorflow_datasets as tfds
from ..dataset_io import rot_net_inputs_and_targets_from_imagenet


def create_model() -> tf.keras.Model:
    """
    Builds the model to use.

    Returns:
        The model that it created.

    """
    model = rotnet_resnet()
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
        imagenet_splits["valid"]
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
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=make_learning_rate(lr_config)
    )
    model.compile(optimizer=optimizer, loss=make_rotnet_loss())
    model.fit(
        training_data,
        validation_data=testing_data,
        epochs=num_epochs,
        validation_freq=validation_frequency,
    )

    return model
