import itertools
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from ...frame_reader import FrameReader
from ...schemas import UnannotatedFeatures as Uf
from ..tfrecords_utils import bytes_feature, int_feature
from ..color_utils import rgb_to_hcl

_FEATURES_TO_FACTORIES = {
    Uf.IMAGE_ENCODED: bytes_feature,
    Uf.IMAGE_SEQUENCE_ID: int_feature,
    Uf.IMAGE_FRAME_NUM: int_feature,
    Uf.IMAGE_HUE_HISTOGRAMS: bytes_feature,
    Uf.IMAGE_CHROMA_HISTOGRAMS: bytes_feature,
}


_SAVED_FRAME_SIZE = (540, 960)
"""
Size to use for the frames in the TFRecord file, in the form of (height, width).
"""


def _compute_histograms(
    image_channel: tf.Tensor,
    window_size: int = 7,
    num_bins: int = 32,
) -> tf.Tensor:
    """
    Computes the pixel-wise histograms of an image, using color information
    from a window around each pixel.

    Args:
        image_channel: The single-channel image to compute histograms for.
            Should be rank 2.
        window_size: The size of the window to use, in pixels.
        num_bins: The number of bins to use for the histogram.

    Returns:
        The histograms it computed. Will be a 3D matrix of shape
        [num_rows, num_columns, num_bins], containing the histograms for each
        input pixel.

    """
    input_shape = tf.shape(image_channel)
    image_channel = tf.expand_dims(tf.expand_dims(image_channel, 0), -1)

    # First, pre-bin all the values in the image.
    binned = tf.histogram_fixed_width_bins(
        image_channel, value_range=[0.0, 1.0], nbins=num_bins, dtype=tf.uint8
    )

    # Extract patches.
    patches = tf.image.extract_patches(
        binned,
        sizes=[1, window_size, window_size, 1],
        strides=[1] * 4,
        rates=[1] * 4,
        padding="SAME",
    )
    # There will be only one batch dimension, so get rid of it. We also
    # flatten everything into a single matrix where each row is a patch.
    num_patches = tf.reduce_prod(patches.shape[:3])
    patches = tf.reshape(patches, tf.stack((num_patches, -1)))

    # Compute histograms of each patch.
    patches_sorted = tf.sort(patches, axis=1)
    histograms = tf.math.bincount(patches_sorted, axis=-1, minlength=num_bins)

    # Reshape to correspond with the image.
    return tf.reshape(histograms, tf.concat((input_shape, [-1]), axis=0))


def _histograms_from_image(
    image: tf.Tensor,
    *,
    histogram_shape: Tuple[int, int, int],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts the hue and chroma histograms from an image.

    Args:
        image: The image to extract histograms from.
        histogram_shape: The size of the histogram to save, in the form
            (height, width, num_bins)

    Returns:
        The hue and chroma histograms.

    """
    # Convert to HCL.
    image_hcl = rgb_to_hcl(image)

    # Make sure the output has the correct shape.
    height, width, num_bins = histogram_shape
    image_hcl_small = tf.image.resize(image_hcl, (height, width))
    # Generate histograms.
    hue_histograms = _compute_histograms(
        image_hcl_small[:, :, 0], num_bins=num_bins
    )
    chroma_histograms = _compute_histograms(
        image_hcl_small[:, :, 1], num_bins=num_bins
    )

    return hue_histograms, chroma_histograms


def _make_example(
    *,
    image: np.ndarray,
    frame_num: int,
    sequence_id: int,
    hue_histogram: np.ndarray,
    chroma_histogram: np.ndarray,
) -> tf.train.Example:
    """
    Creates a TF `Example` for a single frame.

    Args:
        image: The compressed image data for the frame.
        frame_num: The frame number.
        sequence_id: The sequence ID.
        hue_histogram: The histograms for the hue channel.
        chroma_histogram: The histograms for the chroma channel.

    Returns:
        The example that it created.

    """
    # Create the feature dictionary.
    features = {
        Uf.IMAGE_ENCODED.value: _FEATURES_TO_FACTORIES[Uf.IMAGE_ENCODED](
            image
        ),
        Uf.IMAGE_SEQUENCE_ID.value: _FEATURES_TO_FACTORIES[
            Uf.IMAGE_SEQUENCE_ID
        ]((sequence_id,)),
        Uf.IMAGE_FRAME_NUM.value: _FEATURES_TO_FACTORIES[Uf.IMAGE_FRAME_NUM](
            (frame_num,)
        ),
        Uf.IMAGE_HUE_HISTOGRAMS.value: _FEATURES_TO_FACTORIES[
            Uf.IMAGE_HUE_HISTOGRAMS
        ](hue_histogram),
        Uf.IMAGE_CHROMA_HISTOGRAMS.value: _FEATURES_TO_FACTORIES[
            Uf.IMAGE_CHROMA_HISTOGRAMS
        ](chroma_histogram),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def _split_clips(
    video: Iterable[np.ndarray], *, clip_length: int
) -> Iterable[List[np.ndarray]]:
    """
    Splits a video into clips of at most a certain length.

    Args:
        video: The input video.
        clip_length: The maximum length of each clip.

    Yields:
        Each clip, one-by-one.

    """
    clip = []
    for frame in video:
        clip.append(frame)

        if len(clip) >= clip_length:
            yield clip
            clip = []

    # If we have any extra, this will be a last, partial clip.
    if clip:
        yield clip


def _generate_clip_examples(
    clip: Iterable[np.ndarray],
    *,
    sequence_id: int,
    frame_shape: Tuple[int, int],
    histogram_shape: Tuple[int, int, int],
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples for a particular clip.

    Args:
        clip: The clip to generate examples for.
        sequence_id: The sequence ID of the clip.
        frame_shape: The shape to use for the saved image frames, in the
            form (height, width).
        histogram_shape: The shape of the histograms to save, in the form
            (height, width, num_bins)

    Yields:
        Example for each frame in the clip.

    """
    logger.info("Generating examples for clip {}.", sequence_id)

    for frame_num, frame in enumerate(clip):
        # Resize the image.
        frame = cv2.resize(frame, frame_shape[::-1])
        # Generate histograms.
        hue_hist, chroma_hist = _histograms_from_image(
            frame, histogram_shape=histogram_shape
        )

        # Encode the frame.
        encoded, frame_jpeg = cv2.imencode(".jpg", frame)
        assert encoded, "Failed to encode image"

        yield _make_example(
            image=frame_jpeg,
            frame_num=frame_num,
            sequence_id=sequence_id,
            hue_histogram=hue_hist.numpy(),
            chroma_histogram=chroma_hist.numpy(),
        )


def _generate_examples(
    video: FrameReader,
    *,
    video_name: str,
    clip_length: int,
    skip_initial_frames: int = 0,
    **kwargs: Any,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """
    Generates TFRecord examples for an entire video, splitting it up into
    clips.

    Args:
        video: The video to generate annotations for.
        video_name: Unique name for the video.
        clip_length: The length of each clip, in frames.
        skip_initial_frames: How many frames to skip at the beginning of the
            video.
        **kwargs: Will be forwarded to `_generate_clip_examples`.

    Returns:
        A dictionary of functions mapping file names to
        generators of TFRecord examples for each clip.

    """

    def _read_and_process_clip(
        _sequence_id: int, _start_frame: int
    ) -> Iterable[tf.train.Example]:
        """
        Reads and generates examples for a particular clip.

        Args:
            _sequence_id: The sequence ID.
            _start_frame: The clip's starting frame.

        Yields:
            The examples for the clip.

        """
        clip = video.read(_start_frame)
        clip = itertools.islice(clip, clip_length)
        yield from _generate_clip_examples(
            clip, sequence_id=_sequence_id, **kwargs
        )

    clips = {}
    for sequence_id, start_frame in enumerate(
        range(skip_initial_frames, video.num_frames, clip_length)
    ):
        clip_name = f"{video_name}_clip_{sequence_id}.tfrecord"
        clips[clip_name] = partial(
            _read_and_process_clip, sequence_id, start_frame
        )

    return clips


def generate_multiple_video_examples(
    videos: Dict[str, Callable[[], Any]],
    **kwargs: Any,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """

    Args:
        videos: The videos to generate examples for.
        **kwargs: Will be forwarded to `_generate_examples`.

    Returns:
        A dictionary mapping clip names to generators that produce examples
        for each clip.

    """
    clips = {}
    for video_name, video in videos.items():
        clips.update(
            _generate_examples(
                video(),
                video_name=video_name,
                **kwargs,
            )
        )

    return clips


def combine_session_examples(
    *args: Any,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """
    Combines examples from a bunch of clips into a single dictionary.

    Args:
        *args: The examples to combine.

    Returns:
        The combined examples.

    """
    combined = {}
    for session_num, session_examples in enumerate(args):
        for clip_name, examples in session_examples.items():
            # Prefix with unique video names.
            combined[f"session_{session_num}_{clip_name}"] = examples

    return combined
