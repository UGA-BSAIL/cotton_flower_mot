"""
Pipeline for loading TFRecords data.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from ..dataset_io import (
    DataAugmentationConfig,
    HeatMapSource,
    colorization_inputs_and_targets_from_dataset,
    inputs_and_targets_from_datasets,
    rot_net_inputs_and_targets_from_dataset,
)
from .nodes import concat_datasets, fix_dataset_length


def create_pipeline(**_):
    # Preset for loading training data.
    load_datasets = partial(
        inputs_and_targets_from_datasets,
        include_frame=True,
        heat_map_source=HeatMapSource.LOAD,
    )
    # Preset for loading testing and validation data that doesn't randomize
    # or interleave clips.
    load_clips = partial(
        inputs_and_targets_from_datasets,
        interleave=False,
        include_empty=True,
        include_frame=True,
        heat_map_source=HeatMapSource.NONE,
        batch_size=1,
    )
    # Common configuration for standard dataset loading.
    loading_config = dict(
        config="model_config",
        batch_size="params:batch_size",
        drop_probability="params:drop_probability",
    )

    return Pipeline(
        [
            node(
                DataAugmentationConfig,
                dict(
                    max_brightness_delta="params:max_brightness_delta",
                    max_hue_delta="params:max_hue_delta",
                    min_contrast="params:min_contrast",
                    max_contrast="params:max_contrast",
                    min_saturation="params:min_saturation",
                    max_saturation="params:max_saturation",
                    flip="params:flip",
                    max_bbox_jitter="params:bbox_jitter_fraction",
                    tracklet_false_positive_rate="params"
                    ":tracklet_false_positive_rate",
                    detection_false_positive_rate="params"
                    ":detection_false_positive_rate",
                    duplicate_rate="params:duplicate_rate",
                ),
                "data_augmentation_config",
            ),
            # Load the datasets.
            node(
                load_datasets,
                dict(
                    raw_datasets="tfrecord_train",
                    augmentation_config="data_augmentation_config",
                    shuffle_buffer_size="params:shuffle_buffer_size",
                    **loading_config
                ),
                "training_data_no_repeats",
            ),
            node(
                fix_dataset_length,
                dict(
                    dataset="training_data_no_repeats",
                    batch_size="params:batch_size",
                    num_dataset_examples="params:num_dataset_examples",
                ),
                "training_data",
            ),
            node(
                load_datasets,
                dict(raw_datasets="tfrecord_test", **loading_config),
                "testing_data",
            ),
            node(
                load_datasets,
                dict(raw_datasets="tfrecord_valid", **loading_config),
                "validation_data",
            ),
            # Create clip datasets.
            node(
                load_clips,
                dict(
                    raw_datasets="tfrecord_test",
                    config="model_config",
                ),
                "testing_data_clips",
            ),
            node(
                load_clips,
                dict(
                    raw_datasets="tfrecord_valid",
                    config="model_config",
                ),
                "validation_data_clips",
            ),
            node(
                rot_net_inputs_and_targets_from_dataset,
                dict(
                    unannotated_dataset="tfrecord_unannotated_no_wheel_combined",
                    config="model_config",
                    batch_size="params:rot_net_batch_size",
                ),
                "rotnet_unannotated_data",
            ),
            node(
                concat_datasets,
                [
                    "tfrecord_unannotated_no_wheel_combined",
                    "tfrecord_unannotated_wheel_combined",
                ],
                "tfrecord_unannotated_all",
            ),
            node(
                colorization_inputs_and_targets_from_dataset,
                dict(
                    unannotated_dataset="tfrecord_unannotated_all",
                    config="model_config",
                    batch_size="params:colorization_batch_size",
                ),
                "colorization_unannotated_data",
            ),
        ]
    )
