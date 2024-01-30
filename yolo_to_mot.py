"""
Helper script that converts YOLO tracking results to the MOT Challenge format.
"""


from pathlib import Path
from typing import Tuple
import re
import io
from collections import OrderedDict
import argparse
import pandas as pd
from src.cotton_flower_mot.schemas import MotAnnotationColumns


_LABEL_FILE_FRAME_NUM_PATTERN = re.compile(r".+_(\d+).txt")
"""
Pattern to use for extracting frame numbers from label file names.
"""


def _gather_yolo_data(label_dir: Path) -> str:
    """
    Gathers the data from all the YOLO label files into a single
    text corpus.

    Args:
        label_dir: The directory containing YOLO label files.

    Returns:
        The contents of all the files, concatenated.

    """
    label_files = sorted(label_dir.glob("*.txt"))

    all_label_data = []
    for label_file in label_files:
        label_text = label_file.read_text()
        label_frame = int(
            _LABEL_FILE_FRAME_NUM_PATTERN.match(label_file.name).group(1)
        )

        for label_line in label_text.split("\n"):
            if label_line.count(" ") < 6:
                # This line does not have an ID, and is therefore an extraneous
                # detection. Skip it.
                continue

            # Add the frame number as the first column.
            all_label_data.append(f"{label_frame} {label_line}")

    return "\n".join(all_label_data)


def _create_mot_df(
    yolo_data: str, *, frame_size: Tuple[int, int]
) -> pd.DataFrame:
    """
    Creates a DataFrame from the YOLO tracking results

    Args:
        yolo_data: The contents of all the YOLO label files.
        frame_size: The size of the video frames, in pixels (w, h).

    Returns:
        A DataFrame containing the tracking results.

    """
    yolo_df = pd.read_csv(
        io.StringIO(yolo_data),
        sep=" ",
        header=None,
        usecols=[0, 2, 3, 4, 5, 7],
        names=["frame", "x", "y", "w", "h", "id"],
    )
    yolo_df["id"] = yolo_df["id"].astype(int)

    # Sort by frame number.
    yolo_df.sort_values(by="frame", inplace=True)

    # Convert bounding boxes to the MOT Challenge format.
    box_min_x = yolo_df["x"] - yolo_df["w"] / 2
    box_min_y = yolo_df["y"] - yolo_df["h"] / 2
    # Convert to pixels.
    box_min_x_px = box_min_x * frame_size[0]
    box_min_y_px = box_min_y * frame_size[1]
    box_width_px = yolo_df["w"] * frame_size[0]
    box_height_px = yolo_df["h"] * frame_size[1]

    # Create the MOT Challenge DataFrame.
    annotation_data = OrderedDict(
        [
            (MotAnnotationColumns.FRAME, yolo_df["frame"]),
            (MotAnnotationColumns.ID, yolo_df["id"]),
            (MotAnnotationColumns.BBOX_X_MIN_PX, box_min_x_px),
            (MotAnnotationColumns.BBOX_Y_MIN_PX, box_min_y_px),
            (MotAnnotationColumns.BBOX_WIDTH_PX, box_width_px),
            (MotAnnotationColumns.BBOX_HEIGHT_PX, box_height_px),
            (MotAnnotationColumns.CONFIDENCE, 1.0),
            # These are for 3D tracking, which we're not doing.
            ("x", -1),
            ("y", -1),
            ("z", -1),
        ]
    )
    return pd.DataFrame(data=annotation_data)


def _make_parser() -> argparse.ArgumentParser:
    """
    Returns:
        The parser to use for CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Convert YOLO tracking results for MOT Challenge format."
    )

    parser.add_argument(
        "label_dir",
        type=Path,
        help="The directory containing YOLO label files.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="The output MOT Challenge file."
    )

    parser.add_argument(
        "-w",
        "--frame-width",
        type=int,
        default=960,
        help="The width of the video frames.",
    )
    parser.add_argument(
        "-t",
        "--frame-height",
        type=int,
        default=540,
        help="The height of the video frames.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()

    yolo_data = _gather_yolo_data(cli_args.label_dir)
    mot_df = _create_mot_df(
        yolo_data, frame_size=(cli_args.frame_width, cli_args.frame_height)
    )

    mot_df.to_csv(cli_args.output, index=False, header=False)


if __name__ == "__main__":
    main()
