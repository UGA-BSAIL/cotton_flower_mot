"""
Model evaluation pipeline.
"""

from kedro.pipeline import Pipeline, node

from .nodes import compute_counts, compute_tracks_for_clip, make_track_videos


def create_pipeline(**kwargs):
    tracking_params = dict(confidence_threshold="params:conf_threshold")

    return Pipeline(
        [
            # # Compute online tracks.
            # node(
            #     compute_tracks_for_clip,
            #     dict(
            #         model="trained_model",
            #         clip_dataset="testing_data_clips",
            #         **tracking_params
            #     ),
            #     "testing_tracks",
            # ),
            # node(
            #     compute_tracks_for_clip,
            #     dict(
            #         model="trained_model",
            #         clip_dataset="validation_data_clips",
            #         **tracking_params
            #     ),
            #     "validation_tracks",
            # ),
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
