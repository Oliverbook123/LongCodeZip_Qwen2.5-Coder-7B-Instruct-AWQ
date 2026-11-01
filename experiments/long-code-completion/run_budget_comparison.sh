#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="$HOME/miniconda3/envs/py39/bin/python"
samples="$repo_root/experiments/long-code-completion/results/awq-full-500-offload/samples.jsonl"
base_dir="$repo_root/experiments/long-code-completion/results"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--migrate-only") ]]; then
    echo "Usage: $0 [--migrate-only]" >&2
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

for mode in line token; do
    legacy_dir="$base_dir/exp2-$mode-offload4"
    outdir="$base_dir/exp2-$mode-offload4-v2"
    migration_args=()
    if [[ -f "$legacy_dir/manifest.json" ]]; then
        migration_args=(--legacy-from "$legacy_dir")
    fi
    bash experiments/long-code-completion/run_staged_awq.sh \
            --samples "$samples" \
            --outdir "$outdir" \
            "${migration_args[@]}" \
            --target-tokens 2048 \
            --fine-ratio 0.5 \
            --fine-budget-mode "$mode" \
            --chunking-mode source \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.84 \
            --cpu-offload-gb 4 "$@"
done

if [[ "${1:-}" == "--migrate-only" ]]; then
    exit 0
fi

"$python_bin" experiments/long-code-completion/compare_budget_allocation.py \
    "$base_dir/exp2-line-offload4-v2" "$base_dir/exp2-token-offload4-v2"
