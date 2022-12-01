"""
Pipeline definition for model training.
"""


import enum

from kedro.pipeline import Pipeline, node

from ..training_utils import set_check_numerics, set_mixed_precision
from .nodes import create_model, prepare_pretrained_encoder, train_model


@enum.unique
class ModelInit(enum.IntEnum):
    """
    How to initialize the model before training.
    """

    IMAGENET = enum.auto()
    """
    Initialize with ImageNet weights.
    """
    ROTNET = enum.auto()
    """
    Initialize with RotNet weights.
    """
    COLORIZATION = enum.auto()
    """
    Initialize with colorization weights.
    """


def create_pipeline(model_init: ModelInit = ModelInit.IMAGENET):
    """
    Args:
        model_init: Specifies how to initialize the model weights.

    Returns:

    """
    if model_init == ModelInit.IMAGENET:
        # This is the default model loading process.
        init_pipeline = Pipeline(
            [node(create_model, "model_config", "initial_model")]
        )
    else:
        # This is the model loading process using a custom pretrained encoder.
        encoder_model = "pretrained_rotnet_model"
        if model_init == ModelInit.COLORIZATION:
            encoder_model = "pretrained_colorization_model"

        init_pipeline = Pipeline(
            [
                node(
                    prepare_pretrained_encoder,
                    dict(
                        encoder=encoder_model,
                        config="model_config",
                        freeze_fraction="params:freeze_fraction",
                    ),
                    "encoder_model",
                ),
                node(
                    create_model,
                    dict(config="model_config", encoder="encoder_model"),
                    "initial_model",
                ),
            ],
        )

    return init_pipeline + Pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
            node(set_mixed_precision, "params:enable_mixed_precision", None),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    testing_data="validation_data",
                    learning_phases="params:learning_phases",
                    loss_params="params:loss_params",
                    heatmap_loss_weight="params:heatmap_loss_weight",
                    geometry_loss_weight="params:geometry_loss_weight",
                    sinkhorn_loss_weight="params:sinkhorn_loss_weight",
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_period="params:histogram_period",
                    update_period="params:update_period",
                    heatmap_size="params:heatmap_size",
                    heatmap_period="params:heatmap_period",
                    num_heatmap_batches="params:num_heatmap_batches",
                    num_heatmap_images="params:num_heatmap_images",
                ),
                ["best_model", "last_model"],
            ),
        ],
    )
