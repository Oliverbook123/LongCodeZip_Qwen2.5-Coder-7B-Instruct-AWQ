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

Results are written to `results/exp2-line` and `results/exp2-token`, followed
by a per-metric comparison. Use `--fine-budget-mode line` or
`--fine-budget-mode token` when running `staged_awq.py` manually.
