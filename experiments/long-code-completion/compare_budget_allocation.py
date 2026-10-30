"""Compare line- and token-weighted fine-grained experiment outputs."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(directory):
    summary = read_json(directory / "summary.json")
    rows = {
        path.stem: read_json(path)
        for path in (directory / "results").glob("*.json")
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("line_dir", type=Path)
    parser.add_argument("token_dir", type=Path)
    args = parser.parse_args()

    line_summary, line_rows = summarize(args.line_dir)
    token_summary, token_rows = summarize(args.token_dir)
    ids = sorted(set(line_rows) & set(token_rows))
    if not ids:
        raise ValueError("No matching per-sample results found")

    print("metric,line,token,token-minus-line")
    for key in ("average_es", "average_em", "aggregate_background_compression_factor",
                "mean_sample_compression_factor"):
        line_value = line_summary[key]
        token_value = token_summary[key]
        print(f"{key},{line_value:.6f},{token_value:.6f},{token_value - line_value:.6f}")

    es_deltas = [token_rows[idx]["es"] - line_rows[idx]["es"] for idx in ids]
    em_deltas = [token_rows[idx]["em"] - line_rows[idx]["em"] for idx in ids]
    print(f"matched_samples,{len(ids)},,")
    print(f"sample_es_delta_mean,{sum(es_deltas) / len(es_deltas):.6f},,")
    print(f"sample_em_delta_mean,{sum(em_deltas) / len(em_deltas):.6f},,")


if __name__ == "__main__":
    main()
