#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
log_dir="$repo_root/experiments/long-code-completion/results/awq-staged"
mkdir -p "$log_dir"
# Keep output visible in the tmux pane while retaining a persistent log.
exec > >(tee -a "$log_dir/run.log") 2>&1
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
for stage in coarse entropy fine; do
    echo "Starting stage: $stage"
    "$HOME/miniconda3/envs/py39/bin/python" -u experiments/long-code-completion/staged_awq.py "$stage" "$@"
    echo "Finished stage: $stage"
done
