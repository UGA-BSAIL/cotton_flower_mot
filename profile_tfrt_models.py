"""
Utility for profiling the performance of TFRT models.
"""


import argparse
from pathlib import Path
from typing import Dict, Callable, Tuple, Any

import cv2
import time

from loguru import logger

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from tqdm import trange


GraphFunc = Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
"""
Type alias for a function that runs the TF graph.
"""


def _get_func_from_saved_model(saved_model_dir: Path) -> Tuple[GraphFunc, Any]:
    """
    Generates a graph function from a saved TFRT model.

    Args:
        saved_model_dir: The saved model directory.

    Returns:
        The graph function, and the saved model.

    """
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING]
    )
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]

    return graph_func, saved_model_loaded


def _get_test_image() -> tf.Tensor:
    """
    Loads a test image.

    Returns:
        The test image.

    """
    image_path = Path(__file__).parent / "test_images" / "flower_example.png"
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)

    return image


def _get_test_tracking_inputs(
    num_tracklets: int = 2, num_detections: int = 2
) -> Dict[str, tf.Tensor]:
    """
    Creates some fake inputs for the tracking model.

    Args:
        num_tracklets: The number of tracklets to simulate.
        num_detections: The number of detections to simulate.

    Returns:
        The test tracking inputs.

    """
    # We always input the maximum number of detections to the tracking model,
    # and limit it using the row lengths.
    detections_boxes = tf.random.uniform(
        shape=(1, 20, 4), minval=0, maxval=1
    )
    tracklets_boxes = tf.random.uniform(
        shape=(1, 20, 4), minval=0, maxval=1
    )
    detections_appearance = tf.random.normal(
        shape=(1, 20, 392)
    )
    tracklets_appearance = tf.random.normal(
        shape=(1, 20, 392)
    )
    detections_row_lengths = tf.constant([[num_detections]], dtype=tf.int32)
    tracklets_row_lengths = tf.constant([[num_tracklets]], dtype=tf.int32)

    return dict(
        detection_appearance_flat=detections_appearance,
        detection_appearance_row_lengths=detections_row_lengths,
        detection_geometry_flat=detections_boxes,
        detection_geometry_row_lengths=detections_row_lengths,
        tracklet_appearance_flat=tracklets_appearance,
        tracklet_appearance_row_lengths=tracklets_row_lengths,
        tracklet_geometry_flat=tracklets_boxes,
        tracklet_geometry_row_lengths=tracklets_row_lengths,
    )


def _profile_model(model: GraphFunc, inputs: Dict[str, tf.Tensor]) -> None:
    """
    Runs the TF graph and profiles the performance.

    Args:
        model: The TF graph function.
        inputs: The input tensors.

    """
    # The first few iterations are often slow.
    logger.debug("Running warm-up phase...")
    for _ in trange(0, 10):
        model(**inputs)

    # Now do the actual profiling.
    logger.info("Profiling...")
    start_time = time.time()
    outputs = {}
    for _ in trange(0, 100):
        outputs = model(**inputs)
    elapsed_time = time.time() - start_time

    logger.info(
        "Total time: {} seconds ({} iters/second)",
        elapsed_time,
        100 / elapsed_time,
    )


def _profile_detector(saved_model_dir: Path) -> None:
    """
    Profiles the performance of the detector model.

    Args:
        saved_model_dir: The saved model directory.

    """
    logger.info("Profiling detector model {}.", saved_model_dir)

    model, _ = _get_func_from_saved_model(saved_model_dir)
    inputs = {"detections_frame": _get_test_image()}
    _profile_model(model, inputs)


def _profile_tracker(saved_model_dir: Path) -> None:
    """
    Profiles the performance of the tracker model.

    Args:
        saved_model_dir: The saved model directory.

    """
    logger.info("Profiling tracker model {}.", saved_model_dir)

    model, _ = _get_func_from_saved_model(saved_model_dir)
    inputs = _get_test_tracking_inputs()
    _profile_model(model, inputs)


def _make_parser() -> argparse.ArgumentParser:
    """
    Returns:
        The parser to use for CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Profile the performance of a TFRT model."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="The saved model directory.",
    )
    return parser


def main() -> None:
    """
    Main function.
    """
    parser = _make_parser()
    cli_args = parser.parse_args()

    _profile_detector(cli_args.model_dir / "detection_model")
    _profile_tracker(cli_args.model_dir / "tracking_model")


if __name__ == "__main__":
    main()
