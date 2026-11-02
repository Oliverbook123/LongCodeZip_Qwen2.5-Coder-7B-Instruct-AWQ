"""Validation and read-only access for shared, pre-normalization block scores."""
import hashlib
import json
import math

from longcodezip import LongCodeZip
from longcodezip.ami import block_values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def query_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_checkpoint(row, sample, entropy, allocations, coarse_sha, entropy_sha):
    if (row.get("schema_version") != 1 or row["source_index"] != sample["source_index"] or row["coarse_sha256"] != coarse_sha or
            row["entropy_sha256"] != entropy_sha or row["method"] != "conditional_ppl" or
            row["query_sha256"] != query_digest(sample["current_function_context"]) or
            row["fine_budget_allocations"] != allocations or
            set(row["functions"]) != set(allocations)):
        raise ValueError(f"AMI checkpoint inputs or budgets differ: {sample['source_index']}")
    for index, allocation in allocations.items():
        cached = row["functions"][index]
        function = entropy["functions"][index]
        blocks = function["chunks"]
        skipped = allocation["individual_fine_ratio"] >= 0.999 or sum(bool(b.strip()) for b in blocks) < 5
        if (cached["function_sha256"] != function["text_sha256"] or
                cached["blocks_sha256"] != digest(blocks) or
                len(cached["block_weights"]) != len(blocks) or
                any(type(weight) is not int or weight < 0 for weight in cached["block_weights"]) or
                sorted(cached["preserved_blocks"]) != sorted(LongCodeZip.preserved_blocks(blocks)) or
                cached["target_tokens"] != allocation["target_tokens"] or
                cached["scoring_skipped"] != skipped or
                not math.isfinite(cached["scoring_seconds"]) or cached["scoring_seconds"] < 0):
            raise ValueError(f"AMI checkpoint block metadata differs: {sample['source_index']}/{index}")
        scores = cached["raw_ami"]
        if skipped:
            if scores is not None:
                raise ValueError("Skipped functions must not contain fabricated AMI scores")
        elif (not isinstance(scores, list) or len(scores) != len(blocks) or
              any(type(value) not in {float, int} for value in scores)):
            raise ValueError("AMI checkpoint score count or types differ")


class CachedBlockScorer:
    def __init__(self, row):
        self.row = row

    def score_blocks(self, blocks, query, method="conditional_ppl", function_index=None):
        if method != self.row["method"] or query_digest(query) != self.row["query_sha256"]:
            raise ValueError("AMI scorer uses a different query or method")
        cached = self.row["functions"].get(str(function_index))
        if cached is None or cached["blocks_sha256"] != digest(blocks) or cached["scoring_skipped"]:
            raise ValueError("AMI scorer uses different or unscored blocks")
        return list(cached["raw_ami"])


def validate_result(row, cache, entropy, mode):
    if row.get("ami_sha256") != digest(cache) or row.get("block_value_mode") != mode:
        raise ValueError("Result uses missing or stale AMI scores or a different value mode")
    details = {str(index): value for index, value in row["function_compressions"].items()}
    if set(details) != set(cache["functions"]):
        raise ValueError("Result has missing per-function AMI details")
    for index, cached in cache["functions"].items():
        actual = details[index]
        for key in ("raw_ami", "block_weights", "target_tokens", "scoring_skipped"):
            if digest(actual[key]) != digest(cached[key]):
                raise ValueError(f"Result has different {key}: {index}")
        if set(actual["preserved_blocks"]) != set(cached["preserved_blocks"]):
            raise ValueError(f"Result has different preserved blocks: {index}")
        selected = actual["selected_blocks"]
        weights = cached["block_weights"]
        if (len(set(selected)) != len(selected) or
                any(type(i) is not int or i < 0 or i >= len(weights) for i in selected) or
                not set(cached["preserved_blocks"]) <= set(selected)):
            raise ValueError(f"Result contains invalid selected blocks: {index}")
        if cached["scoring_skipped"]:
            if set(selected) != set(range(len(weights))):
                raise ValueError("Skipped functions must keep every block")
            continue
        values, normalization = block_values(entropy["functions"][index]["chunks"], cached["raw_ami"], mode)
        if actual["block_values"] != values or actual["normalization"] != normalization:
            raise ValueError(f"Result has different normalized values: {index}")
        preserved_weight = sum(weights[i] for i in cached["preserved_blocks"])
        remaining = max(0, cached["target_tokens"] - preserved_weight)
        solver = ("preserved_only" if remaining == 0 else
                  "dp" if len(weights) - len(cached["preserved_blocks"]) <= 100 and remaining <= 2000 else "greedy")
        if actual["solver"] != solver:
            raise ValueError(f"Result uses a different solver: {index}")
        if sum(weights[i] for i in selected) > max(cached["target_tokens"], preserved_weight):
            raise ValueError(f"Result exceeds the block selection budget: {index}")
