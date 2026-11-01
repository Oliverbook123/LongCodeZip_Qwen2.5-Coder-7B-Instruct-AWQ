import math
import unittest
from unittest.mock import Mock

import numpy as np

from longcodezip import EntropyChunking


class ChunkingRulesTests(unittest.TestCase):
    def setUp(self):
        self.chunker = EntropyChunking.__new__(EntropyChunking)
        self.text = "def f():\n    first()\n    second()\n    return 1"
        self.values = [1.0, 5.0, 1.0, 1.0]
        self.chunker.calculate_sentence_ppl = Mock(return_value=self.values)

    def test_paper_threshold_uses_standard_deviation_only(self):
        _, threshold = self.chunker.find_ppl_spikes_adaptive(
            self.values, k=0.2, chunking_mode="paper")
        self.assertTrue(math.isclose(threshold, 0.2 * np.std(self.values)))

    def test_actual_chunker_places_peak_at_paper_block_start(self):
        source = self.chunker.chunk_text_adaptive(self.text)
        paper = self.chunker.chunk_text_adaptive(self.text, chunking_mode="paper")
        self.assertEqual(source[0], ["def f():\n    first()", "    second()\n    return 1"])
        self.assertEqual(paper[0], ["def f():", "    first()\n    second()\n    return 1"])
        self.assertEqual(source[3], paper[3])

    def test_cached_scores_produce_the_same_chunks(self):
        for mode in ("source", "paper"):
            with self.subTest(mode=mode):
                actual = self.chunker.chunk_text_adaptive(self.text, chunking_mode=mode)
                self.assertEqual(actual, self.chunker.chunk_from_ppls(actual[1], actual[2], chunking_mode=mode))

    def test_invalid_mode_and_mismatched_score_count_are_rejected(self):
        with self.assertRaises(ValueError):
            self.chunker.chunk_from_ppls(["a"], [], chunking_mode="paper")
        with self.assertRaises(ValueError):
            self.chunker.chunk_text_adaptive(self.text, chunking_mode="invalid")

    def test_blank_lines_and_nonfinite_scores_survive_cache_rebuild(self):
        text = "def f():\n\n    first()\n    return 1"
        sentences = self.chunker.split_into_sentences(text)
        values = [float("inf")] + [1.0] * (len(sentences) - 1)
        self.chunker.calculate_sentence_ppl.return_value = values
        actual = self.chunker.chunk_text_adaptive(text, chunking_mode="paper")
        self.assertEqual(actual, self.chunker.chunk_from_ppls(sentences, values, chunking_mode="paper"))
