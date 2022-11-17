"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .association import AssociationLayer
from .centernet import CenterSizes, ReductionStages
from .dense import DenseBlock, TransitionLayer
from .dla import (
    AggregationNode,
    BasicBlock,
    BottleneckBlock,
    HdaStage,
    UpSamplingIda,
)
from .gnn import DynamicEdgeGcn, ResidualCensNet
from .mlp_conv import MlpConv
from .pooling import PeakLayer
from .utility import BnActConv, BnActDense
from .future.convnext import LayerScale
from .pooling import RoiPooling

# Make sure that Kedro is aware of custom layers.
CUSTOM_LAYERS = {
    layer_class.__name__: layer_class
    for layer_class in (
        MlpConv,
        DenseBlock,
        TransitionLayer,
        AssociationLayer,
        BnActConv,
        BnActDense,
        DynamicEdgeGcn,
        ResidualCensNet,
        AggregationNode,
        BasicBlock,
        HdaStage,
        UpSamplingIda,
        PeakLayer,
        BottleneckBlock,
        ReductionStages,
        CenterSizes,
        LayerScale,
        RoiPooling,
    )
}
if "custom_objects" not in TensorFlowModelDataset.DEFAULT_LOAD_ARGS:
    TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = {}
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"].update(
    CUSTOM_LAYERS
)
