"""
Utility for converting saved models to TFRT. This is meant to be run on
the Jetson.
"""


from functools import partial
from pathlib import Path
from typing import List, Iterable, Callable, Union, Tuple, Optional
import argparse

from loguru import logger
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf


InputFunction = Callable[[], Iterable[List[Union[np.array, tf.Tensor]]]]
"""
Type alias for a function that returns fake inputs to a model.
"""

_MAX_MEMORY = 9000
"""
Maximum memory usage to allow for TF, in MB.
"""


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # The Jetson has unified memory, so if we let TF gobble up all the GPU
    # memory like it wants to by default, that leaves nothing for the CPU.
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_MAX_MEMORY)])


def _generate_detector_inputs(
    batch_size: int = 1, *, input_shape: Tuple[int, int]
) -> InputFunction:
    """
    Generates fake inputs for the detector model.

    Args:
        batch_size: The batch size to use.
        input_shape: The input shape of the model, in terms of (rows, cols).

    Returns:
        The input function for the detector.

    """
    batch_shape = (batch_size,) + input_shape + (3,)
    images = np.random.randint(0, 255, size=batch_shape).astype(np.float32)

    def _input_fn() -> Iterable[List[np.array]]:
        yield [images]

    return _input_fn


def _generate_tracker_inputs(
    *,
    num_appearance_features: int,
    max_detections: int = 20,
) -> InputFunction:
    """
    Generates fake inputs for the tracker model.

    Args:
        num_appearance_features: The number of appearance features we expect.
        max_detections: The maximum number of detections we will allow for a
            single frame.

    Returns:
        The input function for the tracker.

    """
    box_shape = (1, max_detections, 4)
    appearance_shape = (1, max_detections, num_appearance_features)

    boxes = np.random.normal(size=box_shape).astype(np.float32)
    appearance = np.random.normal(size=appearance_shape).astype(np.float32)
    row_lengths = np.random.randint(
        1, max_detections, size=(1, 1), dtype=np.int32
    )

    def _input_fn() -> Iterable[List[tf.Tensor]]:
        # Create dummy values for both detection and tracklet appearances
        # and bounding boxes.
        yield [
            appearance,
            row_lengths,
            boxes,
            row_lengths,
            appearance,
            row_lengths,
            boxes,
            row_lengths,
        ]

    return _input_fn


def _convert_saved_model(
    input_dir: Path,
    output_dir: Path,
    *,
    input_function: InputFunction,
    dynamic_shapes: bool = True,
) -> None:
    """
    Converts a model to TRT.

    Args:
        input_dir: The saved model directory of the input model.
        output_dir: The saved model directory of the output model.
        input_function: Fake inputs to the model that we can use for building
            TRT engines.
        dynamic_shapes: Whether to enable dynamic shapes.

    Returns:

    """
    logger.info("Converting model {}.", input_dir)
    converter_factory = trt.TrtGraphConverterV2
    if dynamic_shapes:
        converter_factory = partial(
            converter_factory,
            use_dynamic_shape=True,
            dynamic_shape_profile_strategy="Optimal",
        )
    converter = converter_factory(
        input_saved_model_dir=input_dir.as_posix(),
        precision_mode=trt.TrtPrecisionMode.FP16,
    )
    converter.convert()
    converter.summary()

    # Build TRT engines.
    logger.debug("Building TRT engines...")
    converter.build(input_fn=input_function)

    # Save the converted model.
    logger.debug("Saving converted model to {}.", output_dir)
    converter.save(output_saved_model_dir=output_dir.as_posix())


def _convert_mot_models(
    *,
    detection_model: Optional[Path],
    tracking_model: Optional[Path],
    output_dir: Path,
    frame_shape: Tuple[int, int],
    num_appearance_features: int,
) -> None:
    """
    Converts the MOT models to TFRT.

    Args:
        detection_model: The saved model directory of the detection model.
        tracking_model: The saved model directory of the tracking model.
        output_dir: The output directory to save the converted models to.
        frame_shape: The shape of the frames in the MOT dataset.
        num_appearance_features: The number of appearance features used by
            the tracking model.

    """
    # Create separate output directories for each model.
    output_dir.mkdir(exist_ok=True)
    detector_output = output_dir / "detection_model"
    tracker_output = output_dir / "tracking_model"

    if detection_model is not None:
        detection_inputs = _generate_detector_inputs(
            batch_size=1, input_shape=frame_shape
        )
        _convert_saved_model(
            input_dir=detection_model,
            output_dir=detector_output,
            input_function=detection_inputs,
            dynamic_shapes=False,
        )

    if tracking_model is not None:
        tracking_inputs = _generate_tracker_inputs(
            num_appearance_features=num_appearance_features
        )
        _convert_saved_model(
            input_dir=tracking_model,
            output_dir=tracker_output,
            input_function=tracking_inputs,
        )
    logger.info("Done converting MOT models.")


def _make_parser() -> argparse.ArgumentParser:
    """
    Creates the parser to use for CLI arguments.

    Returns:
        The parser that it created.

    """
    parser = argparse.ArgumentParser(
        description="Convert an MOT model to TFRT."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("converted_models"),
        help="The directory to write converted models to.",
    )

    parser.add_argument(
        "-d",
        "--detection-model",
        type=Path,
        help="The saved model directory of the detection model.",
    )
    parser.add_argument(
        "-r",
        "--frame-rows",
        type=int,
        default=540,
        help="The height of the input frames that the detector expects.",
    )
    parser.add_argument(
        "-c",
        "--frame-cols",
        type=int,
        default=960,
        help="The width of the input frames that the detector expects.",
    )

    parser.add_argument(
        "-t",
        "--tracking-model",
        type=Path,
        help="The saved model directory of the tracking model.",
    )
    parser.add_argument(
        "-a",
        "--appearance-features",
        type=int,
        default=392,
        help="The number of appearance features the tracker expects.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()

    _convert_mot_models(
        detection_model=cli_args.detection_model,
        tracking_model=cli_args.tracking_model,
        output_dir=cli_args.output,
        frame_shape=(cli_args.frame_rows, cli_args.frame_cols),
        num_appearance_features=cli_args.appearance_features,
    )


if __name__ == "__main__":
    main()
