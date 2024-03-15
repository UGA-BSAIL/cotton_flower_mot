"""
Pipeline for generating model configuration.
"""

from kedro.pipeline import Pipeline, node

from .nodes import make_model_config


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                make_model_config,
                dict(
                    image_input_shape="params:image_input_shape",
                    detection_model_input_shape="params"
                    ":detection_model_input_shape",
                    raw_yolo_input_shape="params:raw_yolo_input_shape",
                    rot_net_input_shape="params:rot_net_input_shape",
                    colorization_input_shape="params:colorization_input_shape",
                    colorization_output_shape="params:colorization_output_shape",
                    frame_input_shape="params:frame_input_shape",
                    num_appearance_features="params:num_appearance_features",
                    num_node_features="params:num_node_features",
                    num_edge_features="params:num_edge_features",
                    sinkhorn_lambda="params:sinkhorn_lambda",
                    num_reduction_stages="params:num_reduction_stages",
                    detection_sigma="params:detection_sigma",
                    nominal_detection_size="params:nominal_detection_size",
                    roi_pooling_size="params:roi_pooling_size",
                ),
                "model_config",
            ),
        ]
    )
