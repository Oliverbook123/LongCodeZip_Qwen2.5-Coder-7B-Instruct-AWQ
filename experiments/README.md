# Experiments

This folder contains the old experiments for the three code-related tasks. Some codes may be outdated after refactoring.

### Quick Start

Each task directory contains a `run.sh` script for easy experimentation. Simply navigate to the desired task directory and run:

```bash
cd <task_directory>
bash run.sh
```

### Code Retrieval (RepoQA)

Navigate to the `repo-qa` directory and run experiments with different compression ratios:

```bash
cd repo-qa
bash run.sh
```

The script will evaluate LongCodeZip on the RepoQA dataset with compression ratios, running experiments in parallel on multiple GPUs.

**Key Parameters:**
- `--compression-ratio`: Controls the compression level
- `--model`: Specifies the base LLM model
- `--backend`: Backend for model inference (vllm)

### Code Completion

Navigate to the `long-code-completion` directory:

```bash
cd long-code-completion
bash run.sh
```

#### Fine-grained budget comparison

The experiment-2 runner evaluates the same fixed 500 samples with the
historical line-weighted allocation and the token-weighted allocation:

```bash
bash run_budget_comparison.sh
```

Results are written to `results/exp2-line-offload4-v2` and
`results/exp2-token-offload4-v2`, followed by a per-metric comparison. Existing
`exp2-*-offload4` checkpoints are validated and migrated automatically; their
original directories are preserved. Migration preserves the historical coarse
choices in each arm; it does not turn experiment 2 into a comparison with shared
coarse choices. Use `--fine-budget-mode line` or
`--fine-budget-mode token` when running `staged_awq.py` manually.

#### Legacy staged scripts

The current staged runner validates and imports legacy data into a separate
directory before resuming. Its default pilot output is `results/awq-staged-v2`,
with `results/awq-staged` as the legacy source. `run_full_awq.sh` writes to
`results/awq-full-500-offload-v2`; the budget runner uses the directories above.
These shell runners use the `py39` environment. Run model inference in tmux.

Check and migrate without loading a model:

```bash
bash run_full_awq.sh --migrate-only
bash run_budget_comparison.sh --migrate-only
bash run_staged_awq.sh --migrate-only
```

For a custom legacy experiment, pass `--legacy-from <old-directory>` and
`--outdir <new-directory>` together with the original samples and model settings.
`staged_awq.py migrate --migrate-from <old-directory>` is the equivalent Python
stage. Complete and partial checkpoints are supported. A migration record
protects against later changes to the legacy source. Verifiable final results
are reused with their original outputs and scores, and completed runs skip model
initialization entirely. Very early outputs without function-budget records are
preserved in their original directories, but only their coarse/entropy caches
are imported; the final stage must be recomputed for those samples. Missing
budget records are never fabricated.

`start_staged_awq.sh` starts the updated staged runner in tmux. Set
`LCZ_STAGED_SESSION` to choose a session name. The historical `run.sh` still
uses its separate `code_compressor.py` implementation; it is not the entry point
for these staged AWQ experiments.

#### Chunking rule comparison

The experiment-3 runner uses the completed `results/exp3-source-offload4`
experiment as a fixed baseline. It imports the exact coarse function choices,
ranking scores, and per-line perplexities into a new paper-rule run. Only the
boundary rule changes; function budgets are checked against the baseline.
Existing experiment directories are preserved.

Prepare and validate all 500 paired caches without loading either model:

```bash
bash run_chunking_comparison.sh --prepare-only
```

Run the paper-rule compression and completion in tmux, then compare results:

```bash
tmux new-session -d -s lcz-exp3-paired -c "$PWD" \
    'bash -o pipefail -c "bash run_chunking_comparison.sh 2>&1 | tee results/exp3-paired-run.log"'
tmux attach -t lcz-exp3-paired
```

Run these commands from `experiments/long-code-completion`. The runner uses
`~/miniconda3/envs/py39/bin/python` and writes new results to
`results/exp3-paper-paired-offload4`, including `comparison.csv` after completion.
Use `LCZ_SOURCE_DIR` and `LCZ_PAPER_DIR` to override the directories.

For manual staged runs, `staged_awq.py prepare --reuse-from <source-directory>`
rebuilds source or paper boundaries directly from cached perplexities, without
model inference. Supply the same `--reuse-from` and experiment options to the
subsequent `fine` command. The manifest records fingerprints of the reference
checkpoints. Missing or stale inputs fail before loading the generation model;
repeating a completed run validates its checkpoints and skips inference.

The original `exp3-paper-offload4` run selected some different coarse functions
and must not be resumed as the paired comparison. Its per-line scores are not
needed for recovery: the completed source run contains all required scores.
Legacy source results can be imported after validation. Legacy intermediate or
final caches without input fingerprints are not silently resumed in place;
use a new output directory for them.

`compare_chunking.py` requires the complete sample set, identical coarse choices
and scores, identical per-line scoring inputs, and identical function budgets.
It recalculates aggregate metrics from validated per-sample results. Rebuilding
boundaries from cached scores is tagged `cached_ppl_rechunking`; this time is
not an entropy-model inference benchmark. New fine results include block
selection details and separate compression/generation times. Historical source
results lack those separate details, so only their combined fine/generation
time is comparable, with normal run-to-run timing variation.

CPU-only regression checks, from the repository root:

```bash
~/miniconda3/envs/py39/bin/python -m unittest discover -s tests -p 'test_*chunking*.py' -v
```
