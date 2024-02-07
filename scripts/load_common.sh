# Load needed modules.
ml python/3.10
ml cuda/12.2.2

# Set this for deterministic runs. For more info, see
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
export PYTHONHASHSEED=0
# Support for TCMalloc.
#export LD_PRELOAD="/apps/eb/gperftools/2.7.90-GCCcore-8.3.0/lib/libtcmalloc_minimal.so.4"
# Support for libdevice.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/compilers/cuda/11.4.3/
