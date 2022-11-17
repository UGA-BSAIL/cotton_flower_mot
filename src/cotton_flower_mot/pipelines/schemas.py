"""
Encapsulates various schemas for the data.
"""


import enum
from typing import Iterable, Type


def _merge_enums(
    merged_name: str, enums: Iterable[Type[enum.Enum]]
) -> Type[enum.Enum]:
    """
    Merges two enums together into one that shares the same members.

    Args:
        merged_name: The name of the merged enum.
        enums: The enums to merge.

    Returns:
        The merged enum class.

    """
    members = []
    for sub_enum in enums:
        members.extend([(e.name, e.value) for e in sub_enum])

    return enum.Enum(value=merged_name, names=members)


@enum.unique
class MotAnnotationColumns(enum.Enum):
    """
    Column names for MOT 1.1 annotations.
    """

    FRAME = "frame"
    """
    The frame number.
    """
    ID = "id"
    """
    The track ID.
    """
    BBOX_X_MIN_PX = "bb_left"
    """
    The minimum x-coordinate of the object bounding box, in pixels.
    """
    BBOX_Y_MIN_PX = "bb_top"
    """
    The minimum y-coordinate of the object bounding box, in pixels.
    """
    BBOX_WIDTH_PX = "bb_width"
    """
    The width of the bounding box, in pixels.
    """
    BBOX_HEIGHT_PX = "bb_height"
    """
    The height of the bounding box, in pixels.
    """
    NOT_IGNORED = "not_ignored"
    """
    Attribute from CVAT specifying whether the annotation is ignored.
    """
    CLASS_ID = "class_id"
    """
    Attribute from CVAT specifying the corresponding label (1-indexed).
    """
    VISIBILITY = "visibility"
    """
    Attribute from CVAT specifying the visibility.
    """


@enum.unique
class ObjectDetectionFeatures(enum.Enum):
    """
    Standard feature names used by the TF Object Detection API. There are
    also additional features here that are specific to our tracking problem.
    """

    IMAGE_HEIGHT = "image/height"
    """
    Height of the image in pixels.
    """
    IMAGE_WIDTH = "image/width"
    """
    Width of the image in pixels.
    """
    IMAGE_FILENAME = "image/filename"
    """
    Filename associated with the image.
    """
    IMAGE_SOURCE_ID = "image/source_id"
    """
    Numerical ID for the image source.
    """
    IMAGE_ENCODED = "image/encoded"
    """
    Encoded image data.
    """
    IMAGE_FORMAT = "image/format"
    """
    Image format.
    """
    OBJECT_BBOX_X_MIN = "image/object/bbox/xmin"
    """
    Min x-value in the detection bounding boxes.
    """
    OBJECT_BBOX_X_MAX = "image/object/bbox/xmax"
    """
    Max x-value in the detection bounding boxes.
    """
    OBJECT_BBOX_Y_MIN = "image/object/bbox/ymin"
    """
    Min y-value in the detection bounding boxes.
    """
    OBJECT_BBOX_Y_MAX = "image/object/bbox/ymax"
    """
    Max y-value in the detection bounding boxes.
    """
    OBJECT_CLASS_TEXT = "image/object/class/text"
    """
    Object class as a human-readable name.
    """
    OBJECT_CLASS_LABEL = "image/object/class/id"
    """
    Object class as a numerical label.
    """

    HEATMAP_ENCODED = "image/heatmap/encoded"
    """
    Encoded image heatmap data. This can be extrapolated easily from existing
    data, but this process is slow, so it's convenient to pre-generate it.
    """


@enum.unique
class _AdditionalTrackingFeatures(enum.Enum):
    """
    Additional features unique to the tracking problem. These are non-standard.
    """

    OBJECT_ID = "image/object/id"
    """
    Unique ID for an object that is used to identify it across multiple frames.
    """
    IMAGE_SEQUENCE_ID = "image/sequence_id"
    """
    Unique ID for identifying frames that belong to the same video clip.
    """
    IMAGE_FRAME_NUM = "image/frame_num"
    """
    The frame number within the clip.
    """


ObjectTrackingFeatures = enum.unique(
    _merge_enums(
        "ObjectTrackingFeatures",
        (ObjectDetectionFeatures, _AdditionalTrackingFeatures),
    )
)
"""
Extends `ObjectDetectionFeatures` with non-standard features unique to the
tracking problem.
"""


@enum.unique
class UnannotatedFeatures(enum.Enum):
    """
    Features for unannotated data.
    """

    IMAGE_ENCODED = ObjectDetectionFeatures.IMAGE_ENCODED.value
    """
    Encoded image data.
    """
    IMAGE_HUE_HISTOGRAMS = "hue_histograms"
    """
    The hue channel pixel-level histograms for the image.
    """
    IMAGE_CHROMA_HISTOGRAMS = "chroma_histograms"
    """
    The chroma channel pixel-level histograms for the image.
    """

    IMAGE_SEQUENCE_ID = ObjectTrackingFeatures.IMAGE_SEQUENCE_ID.value
    """
    Unique ID for identifying frames that belong to the same video clip.
    """
    IMAGE_FRAME_NUM = ObjectTrackingFeatures.IMAGE_FRAME_NUM.value
    """
    The frame number within the clip.
    """


@enum.unique
class ModelInputs(enum.Enum):
    """
    Key names for the inputs to the model.
    """

    DETECTIONS_FRAME = "detections_frame"
    """
    The raw frame image associated with the detections.
    """
    TRACKLETS_FRAME = "tracklets_frame"
    """
    The raw frame image associated with the tracklets.
    """

    DETECTION_GEOMETRY = "detection_geometry"
    """
    Corresponding geometric features for the detection crops. Should have the
    form `[center_x, center_y, width, height]`. These are only used during 
    training. During inference, the model will automatically use the results 
    from the detector.
    """
    TRACKLET_GEOMETRY = "tracklet_geometry"
    """
    Corresponding geometric features for the tracklet crops. Should have the
    form `[center_x, center_y, width, height]`.
    """

    SEQUENCE_ID = "sequence_id"
    """
    The sequence ID of the clip.
    """

    USE_GT_DETECTIONS = "use_gt_detections"
    """
    This is a single boolean input. If true, it will use the DETECTION_GEOMETRY 
    input to supply the ROIs for tracking. If false, the DETECTION_GEOMETRY 
    input will be ignored, and it will instead use the detection results to 
    supply the ROIs.
    """
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    """
    This is a single float input. It supplies a confidence threshold to use 
    for detections. Any detections with lower confidence than this will be 
    discarded. This is mostly applicable when running in inference mode, 
    as it allows the user to control which detections actually make it to 
    the tracker.
    """


@enum.unique
class ModelTargets(enum.Enum):
    """
    Key names for the model targets.
    """

    SINKHORN = "sinkhorn"
    """
    The ground-truth Sinkhorn matrix.
    """
    ASSIGNMENT = "assignment"
    """
    The hard assignment matrix.
    """

    HEATMAP = "heatmap"
    """
    The heatmap indicating the location of the detection centers.
    """
    GEOMETRY_DENSE_PRED = "geometry_dense_pred"
    """
    The geometric features that specify the detection targets. Each feature
    should have the form
    `[center_x, center_y, width, height, offset_x, offset_y]` in the
    ground-truth data. In the predictions, these will instead be in dense
    form, with a vector of `[width, height, offset_x, offset_y]` at each
    pixel location.
    """
    GEOMETRY_SPARSE_PRED = "geometry_sparse_pred"
    """
    Contains the actual detected bounding boxes. For the ground-truth,
    they should be the same as `GEOMETRY_DENSE_PRED`. However, for the
    predictions, this one is in sparse form, i.e. the predictions will have
    the form `[center_x, center_y, width, height, confidence]`.
    """


@enum.unique
class RotNetTargets(enum.Enum):
    """
    Key names for the RotNet model targets.
    """

    ROTATION_CLASS = "rotation_class"
    """
    The predicted rotation class for an input.
    """


@enum.unique
class ColorizationTargets(enum.Enum):
    """
    Key names for the colorization model targets.
    """

    CHROMA_HIST = "chroma_histogram"
    """
    The predicted chroma histogram.
    """
    HUE_HIST = "hue_histogram"
    """
    The predicted hue histogram.
    """
