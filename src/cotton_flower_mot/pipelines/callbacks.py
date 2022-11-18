"""
Encapsulates custom callbacks to use.
"""


import abc
import gc
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks
from loguru import logger

from .schemas import ModelInputs, ModelTargets
from .visualization import visualize_heat_maps
from .model_training.layers import CUSTOM_LAYERS


class _TensorboardLoggingCallback(callbacks.Callback):
    """
    Superclass for callbacks that log to TensorBoard.
    """

    def __init__(self, *, log_dir: Union[str, Path]):
        """
        Args:
            log_dir: The directory to write output logs to.
        """
        super().__init__()

        # Create the SummaryWriter to use.
        self.__writer = tf.summary.create_file_writer(Path(log_dir).as_posix())

    @property
    def _writer(self) -> tf.summary.SummaryWriter:
        return self.__writer


class _ImageLoggingCallback(_TensorboardLoggingCallback, abc.ABC):
    """
    Superclass for callbacks that log images or data derived from processed
    images.
    """

    def __init__(
        self,
        *args: Any,
        model: keras.Model,
        dataset: tf.data.Dataset,
        log_period: int = 1,
        num_images_per_batch: int = 3,
        max_num_batches: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            *args: Will be forwarded to the superclass.
            log_dir: The directory to write output logs to.
            model: The model to run.
            log_period: Specifies that we want to log every this many epochs.
                This can be increased if logging is getting too expensive.
            num_images_per_batch: Maximum number of images to log for each
                batch in the dataset.
            max_num_batches: The maximum number of batches to log data from. If
                not specified, it will log from all of them.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.__model = model
        self.__dataset = dataset
        self.__log_period = log_period
        self.__num_images_per_batch = num_images_per_batch

        # Filter all inputs except the frame.
        self.__dataset = self.__dataset.map(
            lambda i, _: {
                ModelInputs.DETECTIONS_FRAME.value: i[
                    ModelInputs.DETECTIONS_FRAME.value
                ]
            },
        )

        if max_num_batches is not None:
            # Limit to a certain number of images per batch.
            self.__dataset = self.__dataset.unbatch().batch(
                self.__num_images_per_batch
            )
            # Limit the dataset to a set number of batches.
            self.__dataset = self.__dataset.take(max_num_batches)

    @property
    def _model(self) -> keras.Model:
        """
        Returns:
            The model that we are using.
        """
        return self.__model

    def _save_image(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves an image to Tensorboard, using the internal `SummaryWriter`.

        Args:
            *args: Will be forwarded to `tf.summary.image`.
            **kwargs: Will be forwarded to `tf.summary.image`.

        """
        with self._writer.as_default(), tf.summary.record_if(True):
            wrote_success = tf.summary.image(
                *args, max_outputs=self.__num_images_per_batch, **kwargs
            )
            # This should never fail because we assign a default summary writer.
            assert wrote_success, "Writing summary images failed unexpectedly."

    @abc.abstractmethod
    def _log_batch(
        self,
        *,
        image_batch: tf.Tensor,
        epoch: int,
        batch_num: int,
    ) -> None:
        """
        Logs data for a single batch.

        Args:
            image_batch: The input image to log data for.
            epoch: The epoch number.
            batch_num: The batch number.

        """

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.__log_period != 0:
            # Skip logging this epoch.
            return
        logger.debug(
            "Logging with {} for epoch {}...", self.__class__.__name__, epoch
        )

        for batch_num, input_batch in enumerate(self.__dataset):
            logger.debug("Logging for batch {}.", batch_num)
            self._log_batch(
                image_batch=input_batch[ModelInputs.DETECTIONS_FRAME.value],
                epoch=epoch,
                batch_num=batch_num,
            )


def _make_heatmap_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Creates a model that extracts only the heatmap.

    Args:
        model: The base model with all the outputs.

    Returns:
        A model that extracts only the heatmap.

    """
    input_layer = model.get_layer(ModelInputs.DETECTIONS_FRAME.value)
    frame_input = input_layer.get_input_at(0)
    activation_layer = model.get_layer(ModelTargets.HEATMAP.value)
    activation_output = activation_layer.get_output_at(0)
    return tf.keras.Model(inputs=[frame_input], outputs=[activation_output])


class LogHeatmaps(_ImageLoggingCallback):
    """
    Callback that logs the predicted heatmaps at the end of the epoch.
    """

    def __init__(
        self,
        *args: Any,
        resize_images: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            *args: Will be forwarded to the superclass.
            resize_images: If provided, will resize the input images to this
                size, while preserving the aspect ratio. Should be in the form
                `(width, height)`.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.__resize_images = resize_images

        self.__extractor = _make_heatmap_extractor(self._model)

    def _log_batch(
        self,
        *,
        image_batch: tf.Tensor,
        epoch: int,
        batch_num: int,
    ) -> None:
        # Retrieve the activations.
        activations = self.__extractor(image_batch)

        if self.__resize_images is not None:
            # Resize the input images.
            image_batch = tf.image.resize(
                image_batch,
                self.__resize_images[::-1],
                preserve_aspect_ratio=True,
            )

        # Visualize the heatmaps.
        visualizations = visualize_heat_maps(
            images=image_batch,
            features=activations,
            # Softmax or sigmoid activation will put everything between 0
            # and 1.
            max_color_threshold=1.0,
        )

        # Save the heatmaps.
        self._save_image(
            f"Heatmaps (Batch {batch_num})",
            visualizations,
            step=epoch,
        )


class ClearMemory(callbacks.Callback):
    """
    Forces garbage collection at the end of each epoch, which can
    decrease overall memory usage.
    """

    def on_epoch_end(self, _: int, __: Optional[Dict] = None) -> None:
        gc.collect()
        keras.backend.clear_session()


class KeepBest(callbacks.Callback):
    """
    A callback that keeps track of the best-performing model, even if it
    wasn't the most recent.
    """

    def __init__(self, metric: str = "val_loss", minimize: bool = True):
        """
        Args:
            metric: The metric to track for determining which model is the
                best.
            minimize: If true, it will attempt to minimize the metric.
                Otherwise, it will attempt to maximize it.

        """
        self.__metric_name = metric

        if minimize:
            self.__best_metric_value = math.inf
            self.__metric_comparator = lambda x, y: x < y
        else:
            self.__best_metric_value = -math.inf
            self.__metric_comparator = lambda x, y: x > y

        # The current best model.
        self.__best_model = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        # Get the metric value.
        metric_value = logs[self.__metric_name]

        if self.__metric_comparator(metric_value, self.__best_metric_value):
            logger.debug(
                "This model is currently the best at epoch {}.", epoch
            )
            self.__best_metric_value = metric_value

            with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
                self.__best_model = tf.keras.models.clone_model(self.model)

            # Copy the weights too.
            assert len(self.__best_model.layers) == len(self.model.layers)
            for layer, layer_copy in zip(
                self.model.layers, self.__best_model.layers
            ):
                layer_copy.set_weights(layer.get_weights())

    @property
    def best_model(self) -> Optional[tf.keras.Model]:
        """
        Returns:
            The current best model, or None if the callback hasn't ever been
            run.

        """
        return self.__best_model
