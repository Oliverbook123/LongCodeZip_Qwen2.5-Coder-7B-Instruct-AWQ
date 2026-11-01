#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
legacy_dir="$repo_root/experiments/long-code-completion/results/awq-full-500-offload"
result_dir="$repo_root/experiments/long-code-completion/results/awq-full-500-offload-v2"
mkdir -p "$result_dir"
trap 'status=$?; echo "Full experiment exited with status: $status"' EXIT
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
python_bin="$HOME/miniconda3/envs/py39/bin/python"
samples="$legacy_dir/samples.jsonl"
if [[ ! -f "$samples" ]]; then
    existing_samples="$repo_root/experiments/long-code-completion/results/awq-full-500/samples.jsonl"
    if [[ -f "$result_dir/samples.jsonl" ]]; then
        samples="$result_dir/samples.jsonl"
    elif [[ -f "$existing_samples" ]]; then
        samples="$existing_samples"
        echo "Using the previously fixed 500 samples"
    else
        echo "Preparing the first 500 eligible examples from the first 5000 test rows"
        "$python_bin" -u experiments/long-code-completion/pilot_awq.py prepare --count 500 --outdir "$result_dir"
        samples="$result_dir/samples.jsonl"
    fi
fi
"$python_bin" - "$samples" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line.strip()]
pilot = Path('experiments/long-code-completion/results/awq-pilot/samples.jsonl')
if len(rows) != 500:
    raise ValueError(f'Expected 500 samples; found {len(rows)}')
if pilot.exists():
    original = [json.loads(line) for line in pilot.read_text(encoding='utf-8').splitlines() if line.strip()]
    if rows[:len(original)] != original:
        raise ValueError('First samples differ from the completed pilot; stop to investigate dataset version')
    print(f'Verified {len(rows)} samples, including the {len(original)} pilot samples', flush=True)
else:
    print(f'Verified {len(rows)} samples; no local pilot samples to compare', flush=True)
PY
migration_args=()
if [[ -f "$legacy_dir/manifest.json" ]]; then
    migration_args=(--legacy-from "$legacy_dir")
fi
bash experiments/long-code-completion/run_staged_awq.sh \
    --samples "$samples" --outdir "$result_dir" "${migration_args[@]}" \
    --target-tokens 2048 --fine-ratio 0.5 --fine-budget-mode line --chunking-mode source \
    --max-model-len 16384 --gpu-memory-utilization 0.84 --cpu-offload-gb 2 "$@"
