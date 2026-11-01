#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
base_dir="$repo_root/experiments/long-code-completion/results"
outdir="$base_dir/awq-staged-v2"
samples="$base_dir/awq-pilot/samples.jsonl"
legacy_dir="$base_dir/awq-staged"
legacy_explicit=0
custom_outdir=0
migrate_only=0
forwarded=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir|--samples|--legacy-from)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --outdir) outdir="$2"; custom_outdir=1 ;;
                --samples) samples="$2" ;;
                --legacy-from) legacy_dir="$2"; legacy_explicit=1 ;;
            esac
            shift 2 ;;
        --migrate-only) migrate_only=1; shift ;;
        --reuse-from|--migrate-from)
            echo "Use --legacy-from for migration, or run_chunking_comparison.sh for paired experiments" >&2
            exit 2 ;;
        *) forwarded+=("$1"); shift ;;
    esac
done
if [[ "$custom_outdir" == 1 && "$legacy_explicit" == 0 ]]; then
    legacy_dir=""
fi
if [[ -n "$legacy_dir" && ( "$legacy_dir" == "$outdir" || "$legacy_dir" -ef "$outdir" ) ]]; then
    echo "Legacy source and output directories must differ" >&2
    exit 2
fi
if [[ "$legacy_explicit" == 1 && ! -f "$legacy_dir/manifest.json" ]]; then
    echo "Missing legacy manifest: $legacy_dir/manifest.json" >&2
    exit 1
fi
mkdir -p "$outdir"
# Keep output visible in the tmux pane while retaining a persistent log.
exec > >(tee -a "$outdir/run.log") 2>&1
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
python_bin="$HOME/miniconda3/envs/py39/bin/python"
common_args=(--outdir "$outdir" --samples "$samples" "${forwarded[@]}")
if [[ -n "$legacy_dir" && -f "$legacy_dir/manifest.json" ]]; then
    "$python_bin" -u experiments/long-code-completion/staged_awq.py migrate \
        --migrate-from "$legacy_dir" "${common_args[@]}"
fi
if [[ "$migrate_only" == 1 ]]; then
    echo "Migration-only check finished; no model inference started"
    exit 0
fi
for stage in coarse entropy fine; do
    echo "Starting stage: $stage"
    "$python_bin" -u experiments/long-code-completion/staged_awq.py "$stage" "${common_args[@]}"
    echo "Finished stage: $stage"
done
