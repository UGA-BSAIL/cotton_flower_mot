"""
Simple utility to convert MOT challenge results from the CVAT flavor
to the "standard" GT flavor.
"""


import argparse
from pathlib import Path

import pandas as pd


def _make_parser() -> argparse.ArgumentParser:
    """
    Returns:
        The parser for CLI arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file in the CVAT format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("gt.txt"),
        help="Path to the output file in the GT format.",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    cli_args = parser.parse_args()

    annotations = pd.read_csv(
        cli_args.input_file,
        names=["frame", "id", "cx", "cy", "w", "h", "class", "viz", "conf"],
    )
    # Drop class ID and visibility columns, as they aren't in the GT.
    annotations.drop(columns=["class", "viz"], inplace=True)
    # TrackEval expects a class column after the confidence. TrackEval
    # expects this to be 1 for the "pedestrian" class.
    annotations["class"] = 1
    # Add the 3D coordinate columns.
    annotations["x"] = annotations["y"] = annotations["z"] = -1

    annotations.to_csv(cli_args.output, index=False, header=False)


if __name__ == "__main__":
    main()
