"""Value transforms for the block knapsack; independent of model inference."""
import math


def block_values(blocks, scores, mode="raw"):
    """Normalize finite, nonempty blocks within one function, including preserved blocks.

    Constant or absent finite scores map to zero in minmax mode. Invalid scores
    retain the historical zero-value policy. Raw mode preserves all finite scores.
    """
    if mode not in {"raw", "minmax"}:
        raise ValueError(f"Unsupported block_value_mode: {mode}")
    if len(blocks) != len(scores):
        raise ValueError("Block and AMI counts differ")
    valid = [bool(block.strip()) and math.isfinite(score) for block, score in zip(blocks, scores)]
    finite = [float(score) for score, eligible in zip(scores, valid) if eligible]
    low, high = (min(finite), max(finite)) if finite else (None, None)
    if mode == "raw":
        values = [float(score) if math.isfinite(score) else 0.0 for score in scores]
    elif low is None or low == high:
        values = [0.0] * len(scores)
    else:
        # Scaling first also avoids overflow for very large, opposite-signed values.
        scale = max(abs(low), abs(high))
        width = high / scale - low / scale
        values = [(score / scale - low / scale) / width if eligible else 0.0
                  for score, eligible in zip(scores, valid)]
    return values, {
        "mode": mode, "scope": "finite_nonempty_function_blocks_including_preserved",
        "minimum": low, "maximum": high, "valid_count": len(finite),
        "invalid_count": sum(not math.isfinite(score) for score in scores),
        "degenerate": low is None or low == high, "equal_score_policy": "zeros",
        "invalid_score_policy": "zero",
    }
