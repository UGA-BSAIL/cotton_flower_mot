"""
Model evaluation pipeline.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_counts,
    compute_tracks_for_clip,
    make_track_videos,
)
from .inference import build_inference_model
from ..training_utils import set_mixed_precision


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
                compute_tracks_for_clip,
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip_dataset="testing_data_clips",
                ),
                "testing_tracks",
            ),
            node(
                compute_tracks_for_clip,
                dict(
                    tracking_model="inference_tracking_model",
                    detection_model="inference_detection_model",
                    clip_dataset="validation_data_clips",
                ),
                "validation_tracks",
            ),
            # Create count reports.
            # node(
            #     compute_counts,
            #     dict(
            #         tracks_from_clips="testing_tracks",
            #         annotations="annotations_pandas",
            #         counting_line_params="counting_line_params",
            #     ),
            #     "count_report_test",
            # ),
            # node(
            #     compute_counts,
            #     dict(
            #         tracks_from_clips="validation_tracks",
            #         annotations="annotations_pandas",
            #         counting_line_params="counting_line_params",
            #     ),
            #     "count_report_valid",
            # ),
            # Create tracking videos.
            node(
                make_track_videos,
                dict(
                    tracks_from_clips="testing_tracks",
                    clip_dataset="testing_data_clips",
                ),
                "tracking_videos_test",
            ),
            node(
                make_track_videos,
                dict(
                    tracks_from_clips="validation_tracks",
                    clip_dataset="validation_data_clips",
                ),
                "tracking_videos_valid",
            ),
        ]
    )
