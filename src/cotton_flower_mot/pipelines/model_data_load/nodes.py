"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Any

import tensorflow as tf
from loguru import logger


def concat_datasets(*args: Any) -> tf.data.Dataset:
    """
    Concatenates the specified datasets.

    Args:
        *args: The datasets to concatenate.

    Returns:
        The concatenated dataset.

    """
    combined = args[0]
    for dataset in args[1:]:
        combined = combined.concatenate(dataset)

    return combined


def fix_dataset_length(
    dataset: tf.data.Dataset, *, batch_size: int, num_dataset_examples: int
) -> tf.data.Dataset:
    """
    Ensures that a dataset remains the expected length by repeating it if
    necessary.

    Args:
        dataset: The dataset to process.
        batch_size: The batch size.
        num_dataset_examples: The total number of examples in the dataset.

    Returns:
        The fixed dataset.

    """
    need_batches = num_dataset_examples // batch_size
    logger.debug("Dataset should have {} batches.", need_batches)

    return dataset.repeat().take(need_batches)
