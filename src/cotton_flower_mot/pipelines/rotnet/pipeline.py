from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model,
    load_datasets,
    set_mixed_precision,
    train_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(set_mixed_precision, "params:use_mixed_precision", None),
            # Load the data.
            node(
                load_datasets,
                "params:imagenet_dir",
                ["rotnet_training_data", "rotnet_testing_data"],
            ),
            # Build the model.
            node(create_model, [], "initial_rotnet_model"),
            node(
                train_model,
                dict(
                    model="initial_rotnet_model",
                    training_data="rotnet_training_data",
                    testing_data="rotnet_testing_data",
                    lr_config="params:learning_rate",
                    num_epochs="params:num_epochs",
                ),
                "trained_rotnet_model",
            ),
        ]
    )
