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

#### Block AMI normalization comparison (experiment 4)

Experiment 4 fixes source chunking, token-weighted function budgets, preserved
blocks, and the historical solver thresholds. It compares `--block-value-mode raw`
(the backward-compatible default) with `--block-value-mode minmax`. All model
commands use `py39`; no additional dependencies are required.

The normalization policy is explicit: within each function, all finite scores
for nonempty blocks participate in min/max statistics, including preserved
blocks. Constant or absent finite scores map to zero. Invalid scores retain the
historical zero-value treatment and do not determine the extrema. These are
experiment conventions, recorded in the per-function `normalization` metadata.
The current small-function rule and DP/greedy switch are retained. Consequently,
degenerate zero-valued instances may behave differently under the two solvers;
the report counts these cases and records the actual solver for each function.

The runner imports the completed `exp3-source-offload4` baseline's coarse and
entropy inputs. The `ami` stage scores active blocks once, without performing
knapsack selection or completion. Both final arms read the same raw AMI cache.
The old baseline's completion outputs are not substituted for the new raw arm.

From `experiments/long-code-completion`, prepare and validate inputs on CPU:

```bash
bash run_ami_comparison.sh --prepare-only
```

Then start a five-sample pilot in tmux, with live output and a persistent log:

```bash
bash start_ami_comparison.sh --pilot 5
tmux attach -t lcz-exp4
```

After reviewing the pilot, run `bash start_ami_comparison.sh` to continue all
500 samples. The pilot writes per-sample outputs but no full-run summary. Its
existing scores and outputs are validated and reused when the cache grows;
completed stages skip model initialization. `LCZ_AMI_SESSION` overrides the
tmux session name. A live session must finish before starting another run.

Outputs under `results/`:

| Directory/file | Contents |
| --- | --- |
| `exp4-shared-offload4/coarse` and `entropy` | Fixed function choices and blocks |
| `exp4-shared-offload4/ami` | Raw per-block scores, token weights, target budgets, preservation sets, scoring time, and input fingerprints |
| `exp4-raw-offload4/results` | Raw-value selections and completions |
| `exp4-minmax-offload4/results` | Min-Max-value selections and completions |
| Each arm's `summary.json` | Full-run ES, EM and both compression-factor summaries |
| `exp4-comparison.csv` | Validated paired metrics, changed selections, degenerate cases and solver counts |
| `exp4-run.log` | Live run log, also visible in tmux |

Each scored function records `raw_ami`, `block_values`, `block_weights`,
`target_tokens`, `preserved_blocks`, `selected_blocks`, normalization metadata,
solver choice, and selection time. Functions skipped by the historical rule
retain all blocks and use `raw_ami: null` rather than fabricated scores.
Each result binds the exact AMI checkpoint hash and the value mode. Query,
block, budget, and cache mismatches fail before generation. Shared scoring cost
is reported once; cache-read timing in each arm is distinct from model scoring.
The compression time in an arm includes cache access, normalization, selection,
and assembly, while completion generation is timed separately. Allocated
budgets and actual token counts are both reported.

Use `LCZ_AMI_BASE_DIR`, `LCZ_AMI_SAMPLES`, `LCZ_AMI_SOURCE_DIR`,
`LCZ_AMI_SHARED_DIR`, `LCZ_AMI_RAW_DIR`, and `LCZ_AMI_MINMAX_DIR` to override
paths. The source manifest must match the supplied samples and settings.
Legacy scripts default a missing `block_value_mode` to `raw`; experiment 1–3
manifests and checkpoints continue to use their prior modes and fingerprints.

Run the chunking and AMI CPU regression checks from the repository root:

```bash
~/miniconda3/envs/py39/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```
