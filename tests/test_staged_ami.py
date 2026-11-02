"""Pair both treatments against one score cache, with real compression and fake model I/O."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments/long-code-completion"
sys.path.insert(0, str(EXPERIMENT_DIR))
import staged_awq as staged
import compare_ami
from longcodezip import LongCodeZip
import test_staged_chunking as chunking_fixture


class StagedAMITests(unittest.TestCase):
    def setUp(self):
        chunking_fixture.SharedChunkingTests.setUp(self)
        self.args.chunking_mode = "source"
        self.args.block_value_mode = "raw"
        self.args.ami_cache = None
        self.args.max_samples = None
        root = self.source.parent
        self.shared = root / "shared"
        self.raw = root / "raw"
        self.minmax = root / "minmax"

    def engine(self, cached=False):
        engine = LongCodeZip.__new__(LongCodeZip)
        engine.get_token_length = lambda text, **kwargs: len(text)
        engine.tokenizer = SimpleNamespace(encode=lambda text: list(text))
        engine.get_ppl = (Mock(side_effect=AssertionError("recomputed shared block scores")) if cached else
                          Mock(side_effect=lambda text, **kwargs: {"ppl": 1. if kwargs.get("condition_mode") == "prefix" else 2.}))
        engine.engine = SimpleNamespace(generate=Mock(return_value=[
            SimpleNamespace(outputs=[SimpleNamespace(text="return 1")])]))
        return engine

    def model_context(self, engine):
        return patch.dict(sys.modules, {
            "demo_awq": SimpleNamespace(VllmCompressor=Mock(return_value=engine)),
            "vllm": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
        })

    def prepare_shared(self):
        staged.setup(self.shared, self.samples_path, self.args)
        staged.stage_prepare(self.shared, self.samples, self.args)

    def score(self):
        engine = self.engine()
        engine._knapsack_block_selection = Mock(side_effect=AssertionError("selected blocks in scoring stage"))
        with self.model_context(engine):
            staged.stage_ami(self.shared, self.samples, self.args)
        engine.engine.generate.assert_not_called()
        self.assertGreater(engine.get_ppl.call_count, 0)
        return engine

    def finish(self, mode):
        args = copy.copy(self.args)
        args.block_value_mode = mode
        args.ami_cache = self.shared
        directory = self.raw if mode == "raw" else self.minmax
        staged.setup(directory, self.samples_path, args)
        staged.stage_prepare(directory, self.samples, args)
        engine = self.engine(cached=True)
        with self.model_context(engine):
            staged.stage_fine(directory, self.samples, args)
        engine.get_ppl.assert_not_called()
        return args, engine

    def test_both_treatments_share_scores_and_record_selection_details(self):
        self.prepare_shared()
        self.score()
        for mode in ("raw", "minmax"):
            _, engine = self.finish(mode)
            engine.engine.generate.assert_called_once()
        raw = staged.read_json(self.raw / "results/1.json")
        normalized = staged.read_json(self.minmax / "results/1.json")
        self.assertEqual(raw["ami_sha256"], normalized["ami_sha256"])
        self.assertEqual(raw["fine_budget_allocations"], normalized["fine_budget_allocations"])
        self.assertEqual(raw["function_compressions"]["0"]["raw_ami"], normalized["function_compressions"]["0"]["raw_ami"])
        self.assertNotEqual(raw["function_compressions"]["0"]["selected_blocks"], normalized["function_compressions"]["0"]["selected_blocks"])
        metrics, counts = compare_ami.compare(self.raw, self.minmax, self.samples)
        self.assertEqual(counts["functions_with_changed_selection"], 1)
        self.assertEqual(metrics["average_es"], (100., 100.))
        self.assertIn("selection_seconds", raw)
        self.assertEqual(raw["scoring_source"], "shared_ami_cache")

    def test_resume_requires_no_models_and_does_not_rescore(self):
        self.prepare_shared()
        self.score()
        args, _ = self.finish("raw")
        with patch.dict(sys.modules, {"demo_awq": SimpleNamespace(VllmCompressor=Mock(side_effect=AssertionError("loaded a model")))}):
            staged.stage_ami(self.shared, self.samples, self.args)
            staged.stage_fine(self.raw, self.samples, args)

    def test_skipped_functions_keep_all_blocks_without_fabricated_scores(self):
        path = self.source / "entropy/1.json"
        entropy = staged.read_json(path)
        cached = entropy["functions"]["0"]
        from longcodezip import EntropyChunking
        chunker = EntropyChunking.__new__(EntropyChunking)
        chunks, _, ppls, spikes = chunker.chunk_from_ppls(cached["sentences"], [1.] * len(cached["sentences"]))
        cached.update(chunks=chunks, ppls=ppls, spike_indices=spikes)
        staged.atomic_json(path, entropy)
        self.prepare_shared()
        engine = self.engine()
        with self.model_context(engine):
            staged.stage_ami(self.shared, self.samples, self.args)
        engine.get_ppl.assert_not_called()
        self.assertIsNone(staged.read_json(self.shared / "ami/1.json")["functions"]["0"]["raw_ami"])
        self.finish("raw")
        self.finish("minmax")
        _, counts = compare_ami.compare(self.raw, self.minmax, self.samples)
        self.assertEqual(counts["scored_functions"], 0)

    def test_changed_scores_invalidate_already_generated_results(self):
        self.prepare_shared()
        self.score()
        args, _ = self.finish("raw")
        path = self.shared / "ami/1.json"
        cache = staged.read_json(path)
        cache["functions"]["0"]["raw_ami"][1] += 1.
        staged.atomic_json(path, cache)
        with self.assertRaisesRegex(ValueError, "stale AMI"):
            staged.stage_fine(self.raw, self.samples, args)

    def test_changed_query_and_missing_score_rows_fail_before_model_loading(self):
        self.prepare_shared()
        self.score()
        args, _ = self.finish("raw")
        sample = {**self.sample, "current_function_context": "different query"}
        with self.assertRaisesRegex(ValueError, "inputs or budgets"):
            staged.stage_fine(self.raw, [sample], args)
        (self.shared / "ami/1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            staged.stage_fine(self.raw, self.samples, args)

    def test_corrupt_weights_are_detected_before_generation(self):
        self.prepare_shared()
        self.score()
        path = self.shared / "ami/1.json"
        cache = staged.read_json(path)
        cache["functions"]["0"]["block_weights"][1] += 1
        staged.atomic_json(path, cache)
        engine = self.engine(cached=True)
        args = copy.copy(self.args)
        args.ami_cache = self.shared
        staged.setup(self.raw, self.samples_path, args)
        staged.stage_prepare(self.raw, self.samples, args)
        with self.model_context(engine), self.assertRaisesRegex(ValueError, "block_weights"):
            staged.stage_fine(self.raw, self.samples, args)
        engine.engine.generate.assert_not_called()

    def add_second_sample(self):
        sample = {**self.sample, "source_index": 2, "id": "fixture2"}
        self.samples.append(sample)
        self.samples_path.write_text("".join(json.dumps(row) + "\n" for row in self.samples))
        manifest = staged.read_json(self.source / "manifest.json")
        manifest["samples_sha256"] = hashlib.sha256(self.samples_path.read_bytes()).hexdigest()
        staged.atomic_json(self.source / "manifest.json", manifest)
        for stage in ("coarse", "entropy", "results"):
            row = staged.read_json(self.source / stage / "1.json")
            row["source_index"] = 2
            if stage == "results": row["id"] = "fixture2"
            staged.atomic_json(self.source / stage / "2.json", row)

    def test_pilot_resumes_to_full_run_as_cache_grows(self):
        self.add_second_sample()
        self.args.max_samples = 1
        self.prepare_shared()
        self.score()
        self.finish("raw")
        self.finish("minmax")
        self.assertFalse((self.raw / "summary.json").exists())
        self.assertFalse((self.minmax / "summary.json").exists())
        first = (self.raw / "results/1.json").read_bytes()
        self.args.max_samples = None
        self.score()
        self.finish("raw")
        self.finish("minmax")
        self.assertEqual(first, (self.raw / "results/1.json").read_bytes())
        self.assertEqual(staged.read_json(self.raw / "summary.json")["count"], 2)
        _, counts = compare_ami.compare(self.raw, self.minmax, self.samples)
        self.assertEqual(counts["matched_samples"], 2)

    def test_comparison_rejects_missing_samples_and_wrong_mode(self):
        self.prepare_shared()
        self.score()
        self.finish("raw")
        self.finish("minmax")
        path = self.minmax / "results/1.json"
        row = staged.read_json(path)
        row["block_value_mode"] = "raw"
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            compare_ami.compare(self.raw, self.minmax, self.samples)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            compare_ami.compare(self.raw, self.minmax, self.samples)

    def test_prepare_only_shell_entry_loads_no_models(self):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LCZ_AMI_BASE_DIR": str(self.source.parent / "logs"),
                       "LCZ_AMI_SAMPLES": str(self.samples_path), "LCZ_AMI_SOURCE_DIR": str(self.source),
                       "LCZ_AMI_SHARED_DIR": str(self.shared), "LCZ_AMI_RAW_DIR": str(self.raw),
                       "LCZ_AMI_MINMAX_DIR": str(self.minmax)}
        completed = subprocess.run(["bash", str(EXPERIMENT_DIR / "run_ami_comparison.sh"), "--prepare-only"],
                                   env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("no model inference started", completed.stdout)
        self.assertTrue((self.minmax / "entropy/1.json").exists())
        self.assertFalse((self.shared / "ami/1.json").exists())
