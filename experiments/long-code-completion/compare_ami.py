"""Validate and compare paired raw/minmax AMI runs backed by the same scores."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ami_cache import validate_checkpoint, validate_result as validate_ami_result
from staged_awq import (ROOT, checkpoint_paths, coarse_fingerprint, common_config,
                        entropy_fingerprint, normalized_config, read_json, sample_id,
                        summarize_rows, validate_entropy, validate_result)


def load(directory, samples, mode):
    config = normalized_config(read_json(directory / "manifest.json"))
    if config["block_value_mode"] != mode or config["chunking_mode"] != "source" or "ami_cache" not in config:
        raise ValueError(f"Expected a cached source-chunking {mode} AMI run: {directory}")
    stages = {stage: {idx: read_json(path) for idx, path in checkpoint_paths(directory, stage, samples).items()}
              for stage in ("coarse", "entropy", "results")}
    cache_dir = Path(config["ami_cache"]["directory"])
    cache_manifest = cache_dir / "manifest.json"
    if hashlib.sha256(cache_manifest.read_bytes()).hexdigest() != config["ami_cache"]["manifest_sha256"]:
        raise ValueError("AMI cache manifest changed")
    caches = {idx: read_json(path) for idx, path in checkpoint_paths(cache_dir, "ami", samples).items()}
    for sample in samples:
        idx = sample_id(sample)
        coarse, entropy, row = (stages[stage][idx] for stage in ("coarse", "entropy", "results"))
        validate_entropy(sample, coarse, entropy, "source")
        validate_result(sample, coarse, entropy, row, config)
        validate_checkpoint(caches[idx], sample, entropy, row["fine_budget_allocations"],
                            coarse_fingerprint(coarse), entropy_fingerprint(entropy))
        validate_ami_result(row, caches[idx], entropy, mode)
    summary = summarize_rows(list(stages["results"].values()))
    saved = read_json(directory / "summary.json")
    if set(saved) != set(summary) or any(not math.isclose(saved[key], value) for key, value in summary.items()):
        raise ValueError(f"Summary differs from actual results: {directory}")
    return config, stages, caches, summary


def compare(raw_dir, minmax_dir, samples):
    raw_config, raw, caches, raw_summary = load(raw_dir, samples, "raw")
    minmax_config, minmax, _, minmax_summary = load(minmax_dir, samples, "minmax")
    if ({key: value for key, value in common_config(raw_config).items() if key != "block_value_mode"} !=
            {key: value for key, value in common_config(minmax_config).items() if key != "block_value_mode"}):
        raise ValueError("AMI experiment settings differ beyond the value mode")
    counts = Counter({key: 0 for key in (
        "matched_functions", "scored_functions", "functions_with_changed_selection",
        "samples_with_changed_selection", "degenerate_functions", "negative_raw_scores",
        "functions_using_dp", "functions_using_greedy", "functions_using_preserved_only",
        "es_improved_samples", "es_worsened_samples", "es_unchanged_samples")})
    counts["matched_samples"] = len(samples)
    for sample in samples:
        idx = sample_id(sample)
        if (coarse_fingerprint(raw["coarse"][idx]) != coarse_fingerprint(minmax["coarse"][idx]) or
                entropy_fingerprint(raw["entropy"][idx]) != entropy_fingerprint(minmax["entropy"][idx]) or
                raw["results"][idx]["fine_budget_allocations"] != minmax["results"][idx]["fine_budget_allocations"]):
            raise ValueError(f"Function/block inputs or budgets differ: {idx}")
        changed_sample = False
        for index, cached in caches[idx]["functions"].items():
            counts["matched_functions"] += 1
            counts["scored_functions"] += not cached["scoring_skipped"]
            if cached["scoring_skipped"]:
                continue
            a, b = (run["results"][idx]["function_compressions"][index] for run in (raw, minmax))
            changed = set(a["selected_blocks"]) != set(b["selected_blocks"])
            counts["functions_with_changed_selection"] += changed
            counts["degenerate_functions"] += b["normalization"]["degenerate"]
            counts["negative_raw_scores"] += sum(score < 0 for score in cached["raw_ami"] if math.isfinite(score))
            if a["solver"] != b["solver"]:
                raise ValueError(f"Solver changed between arms: {idx}/{index}")
            counts["functions_using_" + a["solver"]] += 1
            changed_sample |= changed
        counts["samples_with_changed_selection"] += changed_sample
        a, b = raw["results"][idx], minmax["results"][idx]
        counts["es_improved_samples"] += b["es"] > a["es"]
        counts["es_worsened_samples"] += b["es"] < a["es"]
        counts["es_unchanged_samples"] += b["es"] == a["es"]
    metrics = {key: (raw_summary[key], minmax_summary[key]) for key in (
        "average_es", "average_em", "aggregate_background_compression_factor", "mean_sample_compression_factor")}
    for key in ("compressed_background_tokens", "compression_seconds", "selection_seconds", "generation_seconds"):
        metrics["total_" + key] = tuple(sum(row[key] for row in run["results"].values()) for run in (raw, minmax))
    metrics["selected_blocks"] = tuple(sum(len(f["selected_blocks"]) for row in run["results"].values()
                                           for f in row["function_compressions"].values()) for run in (raw, minmax))
    counts["shared_scoring_seconds_once"] = sum(f["scoring_seconds"] for row in caches.values() for f in row["functions"].values())
    counts["shared_ami_stage_seconds_once"] = sum(row["seconds"] for row in caches.values())
    return metrics, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("minmax_dir", type=Path)
    parser.add_argument("--samples", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-full-500-offload/samples.jsonl")
    args = parser.parse_args()
    data = args.samples.read_bytes()
    samples = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if not samples or len({sample_id(sample) for sample in samples}) != len(samples):
        raise ValueError("Samples are empty or contain duplicate IDs")
    for directory in (args.raw_dir, args.minmax_dir):
        if read_json(directory / "manifest.json")["samples_sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError("Experiment uses different samples")
    metrics, counts = compare(args.raw_dir, args.minmax_dir, samples)
    print("metric,raw,minmax,minmax-minus-raw")
    for key, (a, b) in metrics.items():
        print(f"{key},{a:.6f},{b:.6f},{b-a:.6f}")
    for key, value in counts.items():
        print(f"{key},{value},,")


if __name__ == "__main__":
    main()
