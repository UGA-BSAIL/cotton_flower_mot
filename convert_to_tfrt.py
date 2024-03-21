"""
Utility for converting saved models to TFRT. This is meant to be run on
the Jetson.
"""


from functools import partial
import itertools
from pathlib import Path
import random
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

_MAX_MEMORY = 7000
"""
Maximum memory usage to allow for TF, in MB.
"""


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # The Jetson has unified memory, so if we let TF gobble up all the GPU
    # memory like it wants to by default, that leaves nothing for the CPU.
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_MAX_MEMORY)],
    )


def _generate_detector_inputs(
    batch_size: int = 1, *, input_shapes: Iterable[Tuple[int, int]]
) -> InputFunction:
    """
    Generates fake inputs for the detector model.

    Args:
        batch_size: The batch size to use.
        input_shapes: The input shapes of the model, in terms of (rows, cols).

    Returns:
        The input function for the detector.

    """
    batch_shapes = [(batch_size,) + s + (3,) for s in input_shapes]
    images = [
        np.random.randint(0, 255, size=s).astype(np.float32)
        for s in batch_shapes
    ]

    def _input_fn() -> Iterable[List[np.array]]:
        for image in images:
            yield [image]

    return _input_fn


def _generate_detector_calibration_inputs(
    batch_size: int = 1,
    *,
    input_shape: Tuple[int, int],
    calibration_images: Path,
) -> InputFunction:
    """
    Generates calibration inputs for the detector model.

    Args:
        batch_size: The batch size to use.
        input_shape: The input shape of the model, in terms of (rows, cols).
        calibration_images: Directory to use containing actual image data to
            use for calibration.

    Returns:
        The input function for the detector.

    """
    # Load random images.
    image_paths = list(calibration_images.glob("*.jpg"))
    logger.debug("Using {} images for calibration.", len(image_paths))
    if not image_paths:
        raise ValueError(
            f"No calibration images found in {calibration_images}"
        )
    random.shuffle(image_paths)
    # Make sure it ends on a multiple of the batch size.
    num_batches = len(image_paths) // batch_size + 1
    image_paths = itertools.islice(
        itertools.cycle(image_paths),
        batch_size * num_batches,
    )

    def _input_fn() -> Iterable[List[tf.Tensor]]:
        for i in range(num_batches):
            batch = []
            for __, image_path in zip(range(batch_size), image_paths):
                # Fill up a batch of images.
                image = tf.io.read_file(image_path.as_posix())
                image = tf.io.decode_image(image, channels=3)

                # Make sure it's the right size.
                if tf.reduce_any(tf.shape(image)[:2] < input_shape):
                    image = tf.image.resize(image, input_shape)
                else:
                    image = tf.image.random_crop(
                        image, input_shape + (3,)
                    )
                    image = tf.cast(image, tf.float32)
                batch.append(image)

            batch = tf.stack(batch)
            logger.debug("Producing batch {} of {}.", i, num_batches)
            yield [batch]

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
    calibration_input_function: Optional[InputFunction] = None,
    dynamic_shapes: bool = True,
) -> None:
    """
    Converts a model to TRT.

    Args:
        input_dir: The saved model directory of the input model.
        output_dir: The saved model directory of the output model.
        input_function: Fake inputs to the model that we can use for building
            TRT engines.
        calibration_input_function: If true, it will convert in INT8
            precision and use this function to provide calibration data.
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
    use_fp16 = calibration_input_function is None
    converter = converter_factory(
        input_saved_model_dir=input_dir.as_posix(),
        precision_mode=trt.TrtPrecisionMode.FP16
        if use_fp16
        else trt.TrtPrecisionMode.INT8,
        use_calibration=not use_fp16,
    )
    converter.convert(calibration_input_fn=calibration_input_function)
    converter.summary()

    # Build TRT engines.
    logger.debug("Building TRT engines...")
    converter.build(input_fn=input_function)

    # Save the converted model.
    logger.debug("Saving converted model to {}.", output_dir)
    converter.save(output_saved_model_dir=output_dir.as_posix())


def _convert_detection_model(
    *,
    model_dir: Path,
    output_dir: Path,
    frame_shape: Tuple[int, int],
    calibration_images: Optional[Path],
) -> None:
    """
    Converts a detection model.

    Args:
        model_dir: The directory containing the saved detection model.
        output_dir: The directory to write the converted detection model to.
        frame_shape: The expected shape of the input images to the model.
        calibration_images: The calibration images for INT8 quantization. If
            not specified, will not use quantization.

    Returns:

    """
    if calibration_images is None:
        logger.info("Using FP16 for the detector model.")

    detection_inputs = _generate_detector_inputs(
        batch_size=1,
        input_shapes=[frame_shape],
    )
    calibration_inputs = None
    if calibration_images is not None:
        calibration_inputs = _generate_detector_calibration_inputs(
            batch_size=1,
            input_shape=frame_shape,
            calibration_images=calibration_images,
        )
    _convert_saved_model(
        input_dir=model_dir,
        output_dir=output_dir,
        input_function=detection_inputs,
        calibration_input_function=calibration_inputs,
        dynamic_shapes=False,
    )


def _convert_mot_models(
    *,
    model_dir: Path,
    output_dir: Path,
    frame_shape: Tuple[int, int],
    small_frame_shape: Tuple[int, int],
    num_appearance_features: int,
    calibration_images: Optional[Path],
) -> None:
    """
    Converts the MOT models to TFRT.

    Args:
        model_dir: The directory containing the saved models.
        output_dir: The output directory to save the converted models to.
        frame_shape: The shape of the frames in the MOT dataset.
        small_frame_shape: The shape of the inputs to the small detector.
        num_appearance_features: The number of appearance features used by
            the tracking model.
        calibration_images: The path to the directory containing calibration
            images. If None, INT8 quantization will be disabled.

    """
    # Create separate output directories for each model.
    output_dir.mkdir(exist_ok=True)
    detector_output = output_dir / "detection_model"
    small_detector_output = output_dir / "small_detection_model"
    tracker_output = output_dir / "tracking_model"

    # Create detection models.
    _convert_detection_model(
        model_dir=model_dir / "detection_model",
        output_dir=detector_output,
        frame_shape=frame_shape,
        calibration_images=calibration_images,
    )
    _convert_detection_model(
        model_dir=model_dir / "small_detection_model",
        output_dir=small_detector_output,
        frame_shape=small_frame_shape,
        calibration_images=calibration_images,
    )

    # Create tracking models.
    tracking_inputs = _generate_tracker_inputs(
        num_appearance_features=num_appearance_features
    )
    _convert_saved_model(
        input_dir=model_dir / "tracking_model",
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
        "model",
        type=Path,
        help="The directory containing the saved models to convert.",
    )
    parser.add_argument(
        "-l",
        "--calibration-images",
        type=Path,
        default=Path("/media/mars/Data/calibration_images/"),
        help="The directory containing calibration images.",
    )
    parser.add_argument(
        "-f",
        "--fp16",
        action="store_true",
        help="Use FP16 instead of INT8 for the detector.",
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
        "--small-frame-rows",
        type=int,
        default=256,
        help="The height of the input frames that the small detector expects.",
    )
    parser.add_argument(
        "--small-frame-cols",
        type=int,
        default=256,
        help="The width of the input frames that the small detector expects.",
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
        model_dir=cli_args.model,
        output_dir=cli_args.output,
        frame_shape=(cli_args.frame_rows, cli_args.frame_cols),
        small_frame_shape=(
            cli_args.small_frame_rows,
            cli_args.small_frame_cols,
        ),
        num_appearance_features=cli_args.appearance_features,
        calibration_images=cli_args.calibration_images
        if not cli_args.fp16
        else None,
    )


if __name__ == "__main__":
    main()
