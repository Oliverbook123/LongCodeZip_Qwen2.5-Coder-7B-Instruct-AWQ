import math
import unittest
from unittest.mock import Mock

from longcodezip import LongCodeZip
from longcodezip.ami import block_values


class AMINormalizationTests(unittest.TestCase):
    def setUp(self):
        self.engine = LongCodeZip.__new__(LongCodeZip)
        self.engine.get_token_length = lambda text, **kwargs: len(text)

    def test_negative_scores_map_to_unit_interval(self):
        values, info = block_values(["a", "b", "c"], [-2., 3., 8.], "minmax")
        self.assertEqual(values, [0., 0.5, 1.])
        self.assertEqual(info["minimum"], -2.)
        self.assertFalse(info["degenerate"])

    def test_constant_empty_and_singleton_scores_have_explicit_zero_policy(self):
        for scores in ([], [3.], [3., 3.], [-2., -2.]):
            with self.subTest(scores=scores):
                values, info = block_values(["block"] * len(scores), scores, "minmax")
                self.assertEqual(values, [0.] * len(scores))
                self.assertTrue(info["degenerate"])
                self.assertEqual(info["equal_score_policy"], "zeros")

    def test_invalid_and_empty_blocks_do_not_set_minmax_extrema(self):
        values, info = block_values(["a", "b", "c", " ", "d"],
                                    [float("nan"), float("inf"), -2., -100., 8.], "minmax")
        self.assertEqual(values, [0., 0., 0., 0., 1.])
        self.assertEqual(info["invalid_count"], 2)
        self.assertEqual(info["valid_count"], 2)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_raw_default_preserves_finite_values_and_historical_invalid_policy(self):
        values, _ = block_values(["a", "b", "c", "d"], [-2., 3., float("inf"), float("nan")])
        self.assertEqual(values, [-2., 3., 0., 0.])

    def test_large_finite_scores_do_not_overflow(self):
        values, _ = block_values(["a", "b", "c"], [-1e308, 0., 1e308], "minmax")
        self.assertEqual(values, [0., 0.5, 1.])

    def test_normalization_changes_the_optimal_combination(self):
        blocks = ["aaaa", "bb", "cc"]
        raw, raw_info = self.engine._knapsack_block_selection(blocks, [8., 5., 5.], 4)
        normalized, info = self.engine._knapsack_block_selection(blocks, [8., 5., 5.], 4,
                                                                 block_value_mode="minmax")
        self.assertEqual(raw, {1, 2})
        self.assertEqual(normalized, {0})
        self.assertEqual(raw_info["solver"], info["solver"])
        self.assertEqual(info["block_values"], [1., 0., 0.])

    def test_preserved_blocks_participate_in_extrema_and_remain_selected(self):
        selected, info = self.engine._knapsack_block_selection(
            ["a", "b", "c"], [-10., 0., 10.], 2, {0}, block_value_mode="minmax")
        self.assertEqual(selected, {0, 2})
        self.assertEqual(info["block_values"], [0., 0.5, 1.])
        selected, info = self.engine._knapsack_block_selection(
            ["aa", "b"], [-1., 2.], 1, {0}, block_value_mode="minmax")
        self.assertEqual(selected, {0})
        self.assertEqual(info["solver"], "preserved_only")

    def test_greedy_threshold_is_the_same_in_both_modes(self):
        for mode in ("raw", "minmax"):
            selected, info = self.engine._knapsack_block_selection(
                ["a"] * 101, list(range(101)), 5, block_value_mode=mode)
            self.assertEqual(info["solver"], "greedy")
            self.assertEqual(len(selected), 5)

    def test_raw_scoring_extraction_keeps_conditional_ppl_semantics(self):
        self.engine.get_ppl = Mock(side_effect=[{"ppl": 5.}, {"ppl": 7.}, {"ppl": 2.}])
        scores = self.engine.score_blocks(["a", " ", "b"], "query")
        self.assertEqual(scores, [-2., -float("inf"), 3.])
        self.assertEqual(self.engine.get_ppl.call_count, 3)

    def test_invalid_modes_and_score_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            block_values(["a"], [], "raw")
        with self.assertRaises(ValueError):
            block_values([], [], "invalid")
        with self.assertRaisesRegex(ValueError, "use_knapsack"):
            self.engine.compress_code_file("def f(): pass", block_value_mode="minmax", use_knapsack=False)
