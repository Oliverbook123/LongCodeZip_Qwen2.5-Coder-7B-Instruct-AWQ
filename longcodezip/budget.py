"""Budget allocation helpers for fine-grained code compression."""

import math
from typing import Dict, Iterable, Mapping, Tuple


def allocate_fine_grained_ratios(
    functions: Iterable[Mapping[str, float]],
    fine_ratio: float,
    min_lines_for_fine_grained: int,
    importance_beta: float,
    budget_mode: str = "line",
) -> Tuple[Dict[int, float], Dict[int, dict]]:
    """Allocate per-function fine-grained retention ratios.

    ``line`` preserves the historical implementation. ``token`` uses the
    tokenizer length as the allocation weight while retaining the existing
    line-based small-function rule.
    """
    if budget_mode not in {"line", "token"}:
        raise ValueError(f"Unsupported fine-grained budget mode: {budget_mode}")
    if not 0.0 <= fine_ratio <= 1.0:
        raise ValueError(f"fine_ratio must be between 0 and 1, got {fine_ratio}")

    rows = [dict(row) for row in functions]
    weights = "line_count" if budget_mode == "line" else "token_count"
    for row in rows:
        if weights not in row:
            raise ValueError(f"Missing {weights} for function {row.get('index')}")

    total_weight = sum(row[weights] for row in rows)
    target_weight = int(total_weight * fine_ratio)
    small = [row for row in rows if row["line_count"] < min_lines_for_fine_grained]
    large = [row for row in rows if row["line_count"] >= min_lines_for_fine_grained]
    small_weight = sum(row[weights] for row in small)
    large_weight = sum(row[weights] for row in large)
    target_large = max(0, target_weight - small_weight)

    ratios = {}
    if not large or large_weight == 0:
        global_rate = 1.0 if large_weight > 0 else 0.0
    elif target_large <= 0:
        global_rate = 0.0
    elif target_large >= large_weight:
        global_rate = 1.0
    else:
        global_rate = target_large / large_weight

    scores = [row.get("score", 0.0) for row in large]
    valid_scores = [score for score in scores if not math.isinf(score) and not math.isnan(score)]
    if large and global_rate not in {0.0, 1.0} and valid_scores and max(valid_scores) != min(valid_scores):
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        normalized_scores = [
            (score - min_score) / (max_score - min_score)
            if not math.isinf(score) and not math.isnan(score) else 0.0
            for score in scores
        ]
        initial_rates = [
            max(0.0, min(1.0, global_rate * (1 + importance_beta * (score - 0.5) * 2)))
            for score in normalized_scores
        ]
        actual_weight = sum(
            initial_rates[index] * row[weights] for index, row in enumerate(large)
        )
        if actual_weight > 0 and abs(actual_weight - target_large) > 1:
            adjustment = target_large / actual_weight
            final_rates = [max(0.0, min(1.0, rate * adjustment)) for rate in initial_rates]
        else:
            final_rates = initial_rates
        ratios.update({row["index"]: final_rates[index] for index, row in enumerate(large)})
    else:
        ratios.update({row["index"]: global_rate for row in large})

    ratios.update({row["index"]: 1.0 for row in small})
    allocation = {
        row["index"]: {
            "line_count": row["line_count"],
            "original_tokens": row["token_count"],
            "allocation_weight": row[weights],
            "target_tokens": int(row["token_count"] * ratios[row["index"]]),
            "individual_fine_ratio": ratios[row["index"]],
            "budget_mode": budget_mode,
        }
        for row in rows
    }
    return ratios, allocation
