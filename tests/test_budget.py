from longcodezip.budget import allocate_fine_grained_ratios


def test_line_mode_uses_line_count_as_allocation_weight():
    functions = [
        {"index": 0, "line_count": 100, "token_count": 1000, "score": 0.0},
        {"index": 1, "line_count": 100, "token_count": 3000, "score": 0.0},
    ]

    ratios, allocations = allocate_fine_grained_ratios(
        functions, 0.5, min_lines_for_fine_grained=5, importance_beta=0.0, budget_mode="line"
    )

    assert ratios == {0: 0.5, 1: 0.5}
    assert allocations[0]["allocation_weight"] == 100
    assert allocations[1]["allocation_weight"] == 100


def test_token_mode_uses_token_count_as_allocation_weight():
    functions = [
        {"index": 0, "line_count": 100, "token_count": 1000, "score": 0.0},
        {"index": 1, "line_count": 100, "token_count": 3000, "score": 0.0},
    ]

    ratios, allocations = allocate_fine_grained_ratios(
        functions, 0.5, min_lines_for_fine_grained=5, importance_beta=0.0, budget_mode="token"
    )

    assert ratios == {0: 0.5, 1: 0.5}
    assert allocations[0]["allocation_weight"] == 1000
    assert allocations[1]["allocation_weight"] == 3000


def test_small_functions_are_kept_in_both_modes():
    functions = [
        {"index": 0, "line_count": 4, "token_count": 1000, "score": 0.0},
        {"index": 1, "line_count": 100, "token_count": 3000, "score": 0.0},
    ]

    ratios, _ = allocate_fine_grained_ratios(
        functions, 0.5, min_lines_for_fine_grained=5, importance_beta=0.0, budget_mode="token"
    )

    assert ratios[0] == 1.0
    assert ratios[1] == 1 / 3
