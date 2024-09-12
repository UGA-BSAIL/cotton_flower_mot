"""
Nodes for the EDA pipeline.
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plot

from ...schemas import ObjectTrackingFeatures as Otf

sns.set()


def annotation_size(annotations: pd.DataFrame) -> plot.Figure:
    """
    Creates a joint histogram of bounding box width and height for the
    annotations.

    Args:
        annotations: The wrangled annotations, in TF format.

    Returns:
        The plot that it created.

    """
    # Calculate width and height.
    x_min = annotations[Otf.OBJECT_BBOX_X_MIN.value]
    x_max = annotations[Otf.OBJECT_BBOX_X_MAX.value]
    y_min = annotations[Otf.OBJECT_BBOX_Y_MIN.value]
    y_max = annotations[Otf.OBJECT_BBOX_Y_MAX.value]

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Plot it.
    axes = sns.jointplot(x=bbox_width, y=bbox_height, kind="hex")

    # Manually adjust the figure size and DPI
    axes.fig.set_size_inches(5, 5)
    axes.fig.set_dpi(300)

    # Set labels and title
    axes.set_axis_labels("Width (px)", "Height (px)", fontsize=12)
    # Adjust 'y' for title position so it doesn't overlap.
    plot.suptitle("Distribution of Bounding Box Sizes", fontsize=14, y=1.02)

    return axes.fig


def annotations_per_frame(annotations: pd.DataFrame) -> plot.Figure:
    """
    Creates a histogram of the number of annotations per frame.

    Args:
        annotations: The wrangled annotations, in TF format.

    Returns:
        The plot that it created.

    """
    plot.figure(figsize=(5, 5), dpi=300)

    # Calculate the number of annotations per frame.
    flowers_per_frame = annotations.pivot_table(
        index=[Otf.IMAGE_SEQUENCE_ID.value, Otf.IMAGE_FRAME_NUM.value],
        aggfunc="size",
    )

    # Plot it.
    sns.histplot(flowers_per_frame, bins=10)
    plot.title("Number of Annotations per Frame", fontsize=14)
    plot.xlabel("Number of Annotations", fontsize=12)
    plot.ylabel("Count", fontsize=12)

    return plot.gcf()


def track_length(annotations: pd.DataFrame) -> plot.Figure:
    """
    Creates a histogram of the track lengths.

    Args:
        annotations: The wrangled annotations, in TF format.

    Returns:
        The plot that it created.

    """
    # Calculate the track lengths.
    track_lengths = annotations.pivot_table(
        index=[Otf.OBJECT_ID.value], aggfunc="size"
    )

    # Plot it.
    axes = sns.displot(track_lengths)
    axes.fig.suptitle("Track Lengths")
    axes.set_axis_labels(xlabel="Number of Frames", ylabel="Count")

    return plot.gcf()
