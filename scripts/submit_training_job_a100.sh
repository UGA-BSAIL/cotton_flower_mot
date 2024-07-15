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
#SBATCH --account=cli2
#SBATCH --qos=cli2
#SBATCH --mail-user=djpetti@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cotton_mot_model_train.%j.out    # Standard output log
#SBATCH --error=cotton_mot_model_train.%j.err     # Standard error log

set -e

source scripts/common.sh
# Prepare the environment.
prepare_environment

# Run the training.
kedro run --pipeline=model_training --env=a100 "$@"
