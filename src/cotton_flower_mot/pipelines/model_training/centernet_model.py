"""
Implementation of the CenterNet detector model.
"""


from functools import partial
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .layers import (
    BnActConv,
    HdaStage,
    PeakLayer,
    ReductionStages,
    TransitionLayer,
    UpSamplingIda,
)
from .layers.resnet import resnet


def _build_backbone(
    normalized_input: tf.Tensor, *, config: ModelConfig
) -> tf.Tensor:
    """
    Builds the backbone for CenterNet, which in this case, is based on DLA.

    Args:
        normalized_input: The normalized input images.
        config: Model configuration.

    Returns:
        The batch of extracted features.

    """
    hda_stage = partial(
        HdaStage,
        agg_filter_size=3,
        activation="relu",
        block_type=HdaStage.Block.BOTTLENECK,
    )

    # Create initial reduction stages.
    reduced_input = ReductionStages(
        num_reduction_stages=config.num_reduction_stages,
    )(normalized_input)

    # Create the main stages of the model.
    hda1 = hda_stage(
        agg_depth=1,
        num_channels=64,
        add_ida_skip=False,
        name="hda_stage_1",
    )(reduced_input)
    transition1 = TransitionLayer()(hda1)
    hda2 = hda_stage(
        agg_depth=2,
        num_channels=128,
        name="hda_stage_2",
    )(transition1)
    transition2 = TransitionLayer()(hda2)
    hda3 = hda_stage(
        agg_depth=2,
        num_channels=256,
        name="hda_stage_3",
    )(transition2)
    transition3 = TransitionLayer()(hda3)
    hda4 = hda_stage(
        agg_depth=1,
        num_channels=512,
        name="hda_stage_4",
    )(transition3)

    # Create the decoder side.
    return UpSamplingIda(
        agg_filter_size=3, activation="relu", name="up_sample"
    )((hda1, hda2, hda3, hda4))


def _build_prediction_head(
    features: tf.Tensor, *, output_channels: int, name: Optional[str] = None
) -> tf.Tensor:
    """
    Builds a prediction head for the model.

    Args:
        features: The features from the backbone.
        output_channels: The number of output channels we want.
        name: Name for the output layer.

    Returns:
        The prediction maps.

    """
    conv1_1 = BnActConv(256, 3, activation="relu", padding="same")(features)
    return BnActConv(output_channels, 1, padding="same", name=name)(conv1_1)


def compute_sparse_predictions(
    *, confidence_masks: tf.Tensor, sizes: tf.Tensor, offsets: tf.Tensor
) -> tf.RaggedTensor:
    """
    Computes sparse predictions given the estimated dense heatmap, sizes,
    and offsets. This is very useful for inference.

    Args:
        confidence_masks: The estimated center point masks, with a positive
            confidence score at the locations of objects and 0 everywhere else,
            with the shape `[batch, height, width, 1]`.
        sizes: The estimated size features, of the form
            `[batch, height, width, 2]`.
        offsets: The estimated offset features, of the form
            `[batch, height, width, 2]`.

    Returns:
        The computed bounding boxes, with the shape
        `[batch, num_boxes, 5]`, where the second dimension is ragged, and the
        third has the form `[center_x, center_y, width, height, confidence]`.

    """
    confidence_masks = tf.ensure_shape(confidence_masks, (None, None, None, 1))
    confidence_masks = confidence_masks[..., 0]
    confidence_masks = tf.cast(confidence_masks, tf.float32)
    center_masks = tf.greater(confidence_masks, 0.0)
    sizes = tf.ensure_shape(sizes, (None, None, None, 2))
    sizes = tf.cast(sizes, tf.float32)
    # offsets = tf.ensure_shape(offsets, (None, None, None, 2))
    # offsets = tf.cast(offsets, tf.float32)
    mask_shape = tf.shape(confidence_masks)[1:3]
    batch_size = tf.shape(confidence_masks)[0]

    # Mask the offsets and sizes.
    sparse_sizes = tf.boolean_mask(sizes, center_masks)
    # sparse_offsets = tf.boolean_mask(offsets, center_masks)
    sparse_confidence = tf.boolean_mask(confidence_masks, center_masks)
    sparse_confidence = tf.expand_dims(sparse_confidence, 1)

    # Figure out the sparse center points.
    sparse_centers = tf.where(center_masks)
    center_points = sparse_centers[:, 1:]
    # Convert to normalized coordinates.
    center_points = tf.cast(center_points, tf.float32) / tf.cast(
        mask_shape, tf.float32
    )

    # Coerce all attributes into a ragged tensor by batch.
    batch_index = sparse_centers[:, 0]
    possible_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1)
    row_lengths = tf.reduce_sum(
        tf.cast(batch_index == possible_indices, tf.int32), axis=1
    )
    ragged_center_points = tf.RaggedTensor.from_row_lengths(
        center_points[..., ::-1], row_lengths
    )
    ragged_sizes = tf.RaggedTensor.from_row_lengths(sparse_sizes, row_lengths)
    # ragged_offsets = tf.RaggedTensor.from_row_lengths(
    #     sparse_offsets, row_lengths
    # )
    ragged_confidence = tf.RaggedTensor.from_row_lengths(
        sparse_confidence, row_lengths
    )

    # Nudge them by the offsets.
    # ragged_center_points += ragged_offsets
    # Add the size.
    return tf.concat(
        (ragged_center_points, ragged_sizes, ragged_confidence), axis=2
    )


def build_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the detection model.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    images = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )

    def _normalize(_images: tf.Tensor) -> tf.Tensor:
        # Normalize the images before putting them through the model.
        float_images = tf.cast(_images, tf.keras.backend.floatx())
        # return tf.image.per_image_standardization(float_images)
        return tf.keras.applications.resnet_v2.preprocess_input(float_images)

    normalized = layers.Lambda(_normalize, name="normalize")(images)

    # Build the model.
    # features = _build_backbone(normalized, config=config)
    features = resnet(image_input=normalized, config=config)
    heatmap = _build_prediction_head(features, output_channels=1)
    sizes = _build_prediction_head(features, output_channels=2)
    offsets = _build_prediction_head(features, output_channels=2)

    # The loss expects sizes and offsets to be merged.
    geometry = layers.Concatenate(
        name=ModelTargets.GEOMETRY_DENSE_PRED.value, dtype=tf.float32
    )((sizes, offsets))
    # Since we treat heatmaps like a pixel-wise classification problem,
    # apply sigmoid activation.
    heatmap = layers.Activation(
        "sigmoid", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(heatmap)

    # Compute sparse bounding boxes for convenience.
    confidence_mask = PeakLayer(with_confidence=True)(heatmap)
    bounding_boxes = layers.Lambda(
        lambda f: compute_sparse_predictions(
            confidence_masks=f[0], sizes=f[1], offsets=f[2]
        ),
        dtype=tf.float32,
        name=ModelTargets.GEOMETRY_SPARSE_PRED.value,
    )((confidence_mask, sizes, offsets))

    return tf.keras.Model(
        inputs=images, outputs=[heatmap, geometry, bounding_boxes]
    )
