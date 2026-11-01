#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="$HOME/miniconda3/envs/py39/bin/python"
samples="$repo_root/experiments/long-code-completion/results/awq-full-500-offload/samples.jsonl"
base_dir="$repo_root/experiments/long-code-completion/results"
source_dir="${LCZ_SOURCE_DIR:-$base_dir/exp3-source-offload4}"
paper_dir="${LCZ_PAPER_DIR:-$base_dir/exp3-paper-paired-offload4}"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--prepare-only") ]]; then
    echo "Usage: $0 [--prepare-only]" >&2
    exit 2
fi

if [[ ! -f "$samples" ]]; then
    echo "Missing fixed 500-sample input: $samples" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ -f "$repo_root/.env" ]]; then
    set -a
    source "$repo_root/.env"
    set +a
fi

common_args=(
    --samples "$samples"
    --outdir "$paper_dir"
    --reuse-from "$source_dir"
    --target-tokens 2048
    --fine-ratio 0.5
    --fine-budget-mode token
    --chunking-mode paper
    --max-model-len 16384
    --gpu-memory-utilization 0.84
    --cpu-offload-gb 4
)

"$python_bin" -u experiments/long-code-completion/staged_awq.py prepare "${common_args[@]}"
if [[ "${1:-}" == "--prepare-only" ]]; then
    exit 0
fi

"$python_bin" -u experiments/long-code-completion/staged_awq.py fine "${common_args[@]}"

"$python_bin" experiments/long-code-completion/compare_chunking.py \
    "$source_dir" "$paper_dir" --samples "$samples" | tee "$paper_dir/comparison.csv"
