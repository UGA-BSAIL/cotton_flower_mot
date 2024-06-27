set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/blue/cli2/$(whoami)/job_scratch/"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/blue/cli2/$(whoami)/mot/"

function prepare_environment() {
  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"

  # Link to the input data directory.
  rm -rf "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/data" "${job_dir}/data"

  # Create output directories.
  mkdir -p "${job_dir}/output_data"
  rm -rf "${job_dir}/logs"
  mkdir -p "${job_dir}/logs"

  # Set the working directory correctly for Kedro.
  cd "${job_dir}"
  # Remove temporary files.
  rm -rf wandb

  # Create the venv.
  ml python/3.10
  if [ ! -d "${SLURM_TMPDIR}/venv" ]; then
    python -m venv "${SLURM_TMPDIR}/venv"
  fi
  rm -f .venv
  ln -s "${SLURM_TMPDIR}/venv" .venv
  source .venv/bin/activate
  pdm sync --prod --no-self
}
