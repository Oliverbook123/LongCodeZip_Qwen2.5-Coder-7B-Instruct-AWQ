#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
base_dir="${LCZ_AMI_BASE_DIR:-$repo_root/experiments/long-code-completion/results}"
python_bin="$HOME/miniconda3/envs/py39/bin/python"
samples="${LCZ_AMI_SAMPLES:-$base_dir/awq-full-500-offload/samples.jsonl}"
source_dir="${LCZ_AMI_SOURCE_DIR:-$base_dir/exp3-source-offload4}"
shared_dir="${LCZ_AMI_SHARED_DIR:-$base_dir/exp4-shared-offload4}"
raw_dir="${LCZ_AMI_RAW_DIR:-$base_dir/exp4-raw-offload4}"
minmax_dir="${LCZ_AMI_MINMAX_DIR:-$base_dir/exp4-minmax-offload4}"
prepare_only=0
pilot_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-only) prepare_only=1; shift ;;
        --pilot)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--pilot requires a positive sample count" >&2; exit 2
            fi
            pilot_args=(--max-samples "$2"); shift 2 ;;
        *) echo "Usage: $0 [--prepare-only] [--pilot N]" >&2; exit 2 ;;
    esac
done
mkdir -p "$base_dir"
exec > >(tee -a "$base_dir/exp4-run.log") 2>&1
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
common=(--samples "$samples" --reuse-from "$source_dir" --target-tokens 2048
        --fine-ratio 0.5 --fine-budget-mode token --chunking-mode source
        --max-model-len 16384 --gpu-memory-utilization 0.84 --cpu-offload-gb 4)
"$python_bin" -u experiments/long-code-completion/staged_awq.py prepare \
    "${common[@]}" --outdir "$shared_dir" --block-value-mode raw
for mode in raw minmax; do
    outdir="$raw_dir"
    if [[ "$mode" == minmax ]]; then outdir="$minmax_dir"; fi
    "$python_bin" -u experiments/long-code-completion/staged_awq.py prepare \
        "${common[@]}" --outdir "$outdir" --block-value-mode "$mode" --ami-cache "$shared_dir"
done
if [[ "$prepare_only" == 1 ]]; then
    echo "Experiment 4 inputs prepared; no model inference started"
    exit 0
fi
"$python_bin" -u experiments/long-code-completion/staged_awq.py ami \
    "${common[@]}" --outdir "$shared_dir" --block-value-mode raw "${pilot_args[@]}"
for mode in raw minmax; do
    outdir="$raw_dir"
    if [[ "$mode" == minmax ]]; then outdir="$minmax_dir"; fi
    "$python_bin" -u experiments/long-code-completion/staged_awq.py fine \
        "${common[@]}" --outdir "$outdir" --block-value-mode "$mode" \
        --ami-cache "$shared_dir" "${pilot_args[@]}"
done
if [[ "${#pilot_args[@]}" -gt 0 ]]; then
    echo "Pilot finished; remove --pilot to resume the full experiment using the same caches"
    exit 0
fi
"$python_bin" experiments/long-code-completion/compare_ami.py "$raw_dir" "$minmax_dir" \
    --samples "$samples" | tee "$base_dir/exp4-comparison.csv"
