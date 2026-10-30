#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="$HOME/miniconda3/envs/py39/bin/python"
samples="$repo_root/experiments/long-code-completion/results/awq-full-500-offload/samples.jsonl"
base_dir="$repo_root/experiments/long-code-completion/results"

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

for mode in line token; do
    outdir="$base_dir/exp2-$mode-offload4"
    for stage in coarse entropy fine; do
        "$python_bin" -u experiments/long-code-completion/staged_awq.py "$stage" \
            --samples "$samples" \
            --outdir "$outdir" \
            --target-tokens 2048 \
            --fine-ratio 0.5 \
            --fine-budget-mode "$mode" \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.84 \
            --cpu-offload-gb 4
    done
done

"$python_bin" experiments/long-code-completion/compare_budget_allocation.py \
    "$base_dir/exp2-line-offload4" "$base_dir/exp2-token-offload4"
