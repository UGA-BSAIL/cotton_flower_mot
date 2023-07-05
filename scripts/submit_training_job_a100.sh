#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=gpu
#SBATCH -J cotton_mot_model_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64gb
#SBATCH --account=lift-phenomics
#SBATCH --qos=lift-phenomics
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cotton_mot_model_train.%j.out    # Standard output log
#SBATCH --error=cotton_mot_model_train.%j.err     # Standard error log

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/blue/lift-phenomics/$(whoami)/job_scratch/"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/blue/lift-phenomics/$(whoami)/mot/"
# Local copy of the dataset.
LOCAL_DATA_DIR="${SLURM_TMPDIR}/data/"

function prepare_environment() {
  # Copy the entire dataset to local scratch.
  echo "Copying dataset..."
  rsync -a "${LARGE_FILES_DIR}/data/"* "${LOCAL_DATA_DIR}"

  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"

  # Link to the input data directory and venv.
  rm -rf "${job_dir}/data"
  ln -s "${LOCAL_DATA_DIR}" "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/.venv" "${job_dir}/.venv"

  # Create output directories.
  mkdir "${job_dir}/output_data"
  mkdir "${job_dir}/logs"

  # Set the working directory correctly for Kedro.
  cd "${job_dir}"
}

# Prepare the environment.
prepare_environment

source scripts/load_common.sh

# Run the training.
poetry run kedro run --pipeline=model_training --env=a100 "$@"
