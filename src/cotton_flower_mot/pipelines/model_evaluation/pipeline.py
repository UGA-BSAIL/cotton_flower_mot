"""
Model evaluation pipeline.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_counts,
    compute_tracks_for_clip_dataset,
    make_track_videos_clip_dataset,
    compute_tracks_for_clip,
    merge_track_datasets,
    make_track_videos_clip,
    filter_countable_tracks,
    make_horizontal_displacement_histogram,
    make_vertical_displacement_histogram,
)
from .inference import build_inference_model
from ..training_utils import set_mixed_precision


_ANALYSIS_SESSIONS = {"2021-08-25_SPL", "2022-08-31_SPL"}
"""
Sessions to perform more advanced analysis on.
"""


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(set_mixed_precision, "params:enable_mixed_precision", None),
            # Create the inference model.
            node(
                build_inference_model,
                dict(
                    training_model="best_model",
                    config="model_config",
                    confidence_threshold="params:conf_threshold",
                    nms_iou_threshold="params:nms_iou_threshold",
                ),
                ["inference_tracking_model", "inference_detection_model"],
            ),
            # Compute online tracks.
            node(
                compute_tracks_for_clip_dataset,
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip_dataset="testing_data_clips",
                ),
                "testing_tracks",
            ),
            node(
                compute_tracks_for_clip_dataset,
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip_dataset="validation_data_clips",
                ),
                "validation_tracks_tfrecord",
            ),
            node(
                partial(compute_tracks_for_clip, name="2022-08-23_ENGR"),
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip="video_2022_08_23_ENGR",
                ),
                "2022-08-23_ENGR_tracks",
            ),
            node(
                partial(compute_tracks_for_clip, name="2021-08-25_SPL"),
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip="video_2021_08_25_SPL",
                ),
                "2021-08-25_SPL_tracks",
            ),
            node(
                partial(compute_tracks_for_clip, name="2022-08-31_SPL"),
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip="video_2022_08_31_SPL",
                ),
                "2022-08-31_SPL_tracks",
            ),
            # Merge all the tracks.
            node(
                merge_track_datasets,
                [
                    "validation_tracks_tfrecord",
                    "2021-08-25_SPL_tracks",
                    "2022-08-23_ENGR_tracks",
                    "2022-08-31_SPL_tracks",
                ],
                "validation_tracks",
            ),
            # Create count reports.
            node(
                filter_countable_tracks,
                dict(
                    tracks_from_clips="testing_tracks",
                    counting_line_params="counting_line_params",
                ),
                "filtered_testing_tracks",
            ),
            node(
                filter_countable_tracks,
                dict(
                    tracks_from_clips="validation_tracks",
                    counting_line_params="counting_line_params",
                ),
                "filtered_validation_tracks",
            ),
            node(
                compute_counts,
                "filtered_testing_tracks",
                "count_report_test",
            ),
            node(
                compute_counts,
                "filtered_validation_tracks",
                "count_report_valid",
            ),
            # Create histograms.
            node(
                lambda tracks: {
                    s: t for s, t in tracks.items() if s in _ANALYSIS_SESSIONS
                },
                "filtered_validation_tracks",
                "analysis_tracks",
            ),
            node(
                make_horizontal_displacement_histogram,
                "analysis_tracks",
                "horizontal_displacement_histogram",
            ),
            node(
                make_vertical_displacement_histogram,
                "analysis_tracks",
                "vertical_displacement_histogram",
            ),
            # Create tracking videos.
            node(
                make_track_videos_clip_dataset,
                dict(
                    tracks_from_clips="testing_tracks",
                    clip_dataset="testing_data_clips",
                    counting_line_params="counting_line_params",
                ),
                "tracking_videos_test",
            ),
            node(
                make_track_videos_clip_dataset,
                dict(
                    tracks_from_clips="validation_tracks",
                    clip_dataset="validation_data_clips",
                    counting_line_params="counting_line_params",
                ),
                "tracking_videos_valid_tfrecord",
            ),
            node(
                partial(make_track_videos_clip, sequence_id="2021-08-25_SPL"),
                dict(
                    tracks_from_clips="validation_tracks",
                    clip="video_2021_08_25_SPL",
                    counting_line_params="counting_line_params",
                ),
                "tracking_video_2021-08-25_SPL",
            ),
            node(
                partial(make_track_videos_clip, sequence_id="2022-08-23_ENGR"),
                dict(
                    tracks_from_clips="validation_tracks",
                    clip="video_2022_08_23_ENGR",
                    counting_line_params="counting_line_params",
                ),
                "tracking_video_2022-08-23_ENGR",
            ),
            node(
                partial(make_track_videos_clip, sequence_id="2022-08-31_SPL"),
                dict(
                    tracks_from_clips="validation_tracks",
                    clip="video_2022_08_31_SPL",
                    counting_line_params="counting_line_params",
                ),
                "tracking_video_2022-08-31_SPL",
            ),
            node(
                merge_track_datasets,
                [
                    # "tracking_videos_valid_tfrecord",
                    # "tracking_video_2021-08-25_SPL",
                    # "tracking_video_2022-08-23_ENGR",
                    "tracking_video_2022-08-31_SPL",
                ],
                "tracking_videos_valid",
            ),
        ]
    )
