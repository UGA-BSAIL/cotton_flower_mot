"""
Pipeline for pre-training the model on the colorization task.
"""

from kedro.pipeline import Pipeline, node, pipeline

from src.cotton_flower_mot.training_utils import set_check_numerics
from .nodes import create_model, train_model


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
            node(create_model, "model_config", "initial_colorization_model"),
            node(
                train_model,
                dict(
                    model="initial_colorization_model",
                    training_data="colorization_unannotated_data",
                    learning_phases="params:colorization_learning_phases",
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_period="params:histogram_period",
                    update_period="params:update_period",
                ),
                "pretrained_colorization_model_output",
            ),
        ]
    )
