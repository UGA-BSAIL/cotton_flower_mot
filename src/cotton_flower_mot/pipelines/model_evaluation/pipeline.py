"""
Model evaluation pipeline.
"""

from kedro.pipeline import Pipeline, node

from .nodes import compute_counts, compute_tracks_for_clip, make_track_videos
from .inference import build_inference_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Create the inference model.
            node(
                build_inference_model,
                dict(
                    training_model="trained_model",
                    config="model_config",
                    confidence_threshold="params:conf_threshold",
                    nms_iou_threshold="params:nms_iou_threshold",
                ),
                "inference_model",
            ),
            # Compute online tracks.
            node(
                compute_tracks_for_clip,
                dict(
                    model="inference_model",
                    clip_dataset="testing_data_clips",
                ),
                "testing_tracks",
            ),
            node(
                compute_tracks_for_clip,
                dict(
                    model="inference_model",
                    clip_dataset="validation_data_clips",
                ),
                "validation_tracks",
            ),
            # Create count reports.
            node(
                compute_counts,
                dict(
                    tracks_from_clips="testing_tracks",
                    annotations="annotations_pandas",
                ),
                "count_report_test",
            ),
            node(
                compute_counts,
                dict(
                    tracks_from_clips="validation_tracks",
                    annotations="annotations_pandas",
                ),
                "count_report_valid",
            ),
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
