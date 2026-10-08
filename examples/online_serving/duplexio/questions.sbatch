#!/usr/bin/env bash
# Run test_inference.sh on one H100 of the DCAI cluster.
# Submit from the repo root: sbatch examples/online_serving/duplexio/questions.sbatch [questions.json] [run_questions.py options...]
# Results: outputs/duplexio-inference/<jobid>/, log: outputs/duplexio-inference/<jobid>.log
#SBATCH --job-name=duplexio-10-questions
#SBATCH --account=iu_0124
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --chdir=/dcai/users/cardua/vllm-omni
#SBATCH --output=/dcai/users/cardua/vllm-omni/outputs/duplexio-inference/%j.log
set -euo pipefail
repo=/dcai/users/cardua/vllm-omni
export PATH=/usr/local/bin:/usr/bin:/bin:/dcai/software/tenant/apps/slurm/current/bin
export HF_HOME=/dcai/users/cardua/.cache/huggingface HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
# Jobs can share a node; keep each server on its own port.
export DUPLEXIO_PORT=${DUPLEXIO_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export PYTHONPATH=$repo
export KOKORO_ASSETS=/opt/kokoro
export TRITON_CACHE_DIR=/var/tmp/cardua/duplexio-validation/triton
export VLLM_CACHE_ROOT=/var/tmp/cardua/duplexio-validation/vllm
export TORCHINDUCTOR_CACHE_DIR=/var/tmp/cardua/duplexio-validation/inductor
questions=${1:-$repo/examples/online_serving/duplexio/questions.json}
shift || true
cd "$repo"
/dcai/software/tenant/apps/Apptainer/1.3.6/usr/bin/apptainer exec --nv --bind /dcai:/dcai \
    /dcai/users/cardua/.cache/apptainer/duplexio-omni-inspect.sif \
    bash examples/online_serving/duplexio/test_inference.sh \
    "$repo/outputs/duplexio-inference/$SLURM_JOB_ID" "$questions" "$@"
