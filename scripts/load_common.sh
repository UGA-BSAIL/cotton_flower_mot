# Load needed modules.
ml Python/3.8.6-GCCcore-10.2.0
ml CUDA/11.2.1
ml cuDNN/8.1.0.77-CUDA-11.2.1
ml gperftools

# Set this for deterministic runs. For more info, see
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
export PYTHONHASHSEED=0
# Support for TCMalloc.
#export LD_PRELOAD="/apps/eb/gperftools/2.7.90-GCCcore-8.3.0/lib/libtcmalloc_minimal.so.4"
# Support for libdevice.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/eb/CUDAcore/11.2.1/
