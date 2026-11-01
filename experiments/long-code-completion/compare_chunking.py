"""Compare complete, paired source and paper chunking experiments."""

import argparse
import hashlib
import json
from pathlib import Path


from staged_awq import (
    ROOT, checkpoint_paths, coarse_fingerprint, common_config, normalized_config,
    read_json, sample_id, summarize_rows, validate_entropy, validate_result,
)


def load_results(directory, samples, mode):
    config = normalized_config(read_json(directory / "manifest.json"))
    if config["chunking_mode"] != mode:
        raise ValueError(f"Expected {mode} experiment: {directory}")
    stages = {stage: {idx: read_json(path) for idx, path in
                     checkpoint_paths(directory, stage, samples).items()}
              for stage in ("coarse", "entropy", "results")}
    for sample in samples:
        idx = sample_id(sample)
        coarse, entropy, row = (stages[stage][idx] for stage in ("coarse", "entropy", "results"))
        validate_entropy(sample, coarse, entropy, mode, allow_legacy=mode == "source")
        validate_result(sample, coarse, entropy, row, config, allow_legacy=mode == "source")
    summary = summarize_rows(list(stages["results"].values()))
    saved = read_json(directory / "summary.json")
    # Summation order can change the last bits of a floating-point average.
    import math
    if set(saved) != set(summary) or any(not math.isclose(saved[key], value) for key, value in summary.items()):
        raise ValueError(f"Summary differs from per-sample results: {directory}")
    return config, summary, stages


def compare(source_dir, paper_dir, samples):
    source_config, source_summary, source = load_results(source_dir, samples, "source")
    paper_config, paper_summary, paper = load_results(paper_dir, samples, "paper")
    if common_config(source_config) != common_config(paper_config):
        raise ValueError("Experiment settings differ beyond chunking mode")
    changed_functions = changed_samples = total_functions = 0
    for sample in samples:
        idx = sample_id(sample)
        if coarse_fingerprint(source["coarse"][idx]) != coarse_fingerprint(paper["coarse"][idx]):
            raise ValueError(f"Coarse selections or scores differ for sample {idx}")
        if source["results"][idx]["fine_budget_allocations"] != paper["results"][idx]["fine_budget_allocations"]:
            raise ValueError(f"Function budgets differ for sample {idx}")
        sample_changed = False
        for index, original in source["entropy"][idx]["functions"].items():
            modified = paper["entropy"][idx]["functions"][index]
            # JSON permits nonfinite PPL values; compare their serialized forms.
            if (original["text_sha256"] != modified["text_sha256"] or
                    original["sentences"] != modified["sentences"] or
                    json.dumps(original["ppls"]) != json.dumps(modified["ppls"])):
                raise ValueError(f"Per-line scoring inputs differ for sample {idx}/{index}")
            changed = original["chunks"] != modified["chunks"]
            changed_functions += changed
            total_functions += 1
            sample_changed |= changed
        changed_samples += sample_changed
    metrics = {key: (source_summary[key], paper_summary[key]) for key in (
        "average_es", "average_em", "aggregate_background_compression_factor", "mean_sample_compression_factor"
    )}
    metrics["total_blocks"] = tuple(sum(len(function["chunks"])
                                       for row in run["entropy"].values()
                                       for function in row["functions"].values()) for run in (source, paper))
    metrics["fine_and_generation_seconds"] = tuple(sum(row["seconds"] for row in run["results"].values())
                                                   for run in (source, paper))
    # Rechunking cached scores is not equivalent to timing entropy-model inference.
    return metrics, {"matched_samples": len(samples), "matched_functions": total_functions,
                     "samples_with_changed_chunks": changed_samples,
                     "functions_with_changed_chunks": changed_functions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("paper_dir", type=Path)
    parser.add_argument("--samples", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-full-500-offload/samples.jsonl")
    args = parser.parse_args()

    data = args.samples.read_bytes()
    samples = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if not samples or len({sample_id(sample) for sample in samples}) != len(samples):
        raise ValueError("Sample list is empty or contains duplicates")
    for directory in (args.source_dir, args.paper_dir):
        if read_json(directory / "manifest.json")["samples_sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Experiment uses different sample data: {directory}")
    metrics, counts = compare(args.source_dir, args.paper_dir, samples)
    print("metric,source,paper,paper-minus-source")
    for key, (original, modified) in metrics.items():
        print(f"{key},{original:.6f},{modified:.6f},{modified - original:.6f}")
    for key, value in counts.items():
        print(f"{key},{value},,")


if __name__ == "__main__":
    main()
