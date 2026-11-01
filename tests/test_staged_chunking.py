"""Exercise shared-cache recovery and the real compression call without models."""
import copy
import hashlib
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments/long-code-completion"
sys.path.insert(0, str(EXPERIMENT_DIR))
import compare_chunking as comparison
import staged_awq as staged
from longcodezip import EntropyChunking, LongCodeZip
from longcodezip.budget import allocate_fine_grained_ratios


class SharedChunkingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.paper = root / "paper"
        self.sample = {"source_index": 1, "id": "fixture", "gt": "return 1",
                       "background_context": "def f():\n" + "\n".join(f"    call_{i}()" for i in range(11)) + "\n    return 1",
                       "current_function_context": "def g():"}
        self.samples = [self.sample]
        self.samples_path = root / "samples.jsonl"
        import json
        self.samples_path.write_text(json.dumps(self.sample) + "\n", encoding="utf-8")
        self.args = SimpleNamespace(target_tokens=2048, fine_ratio=0.5, fine_budget_mode="token",
                                    chunking_mode="source", max_model_len=16384,
                                    gpu_memory_utilization=0.84, cpu_offload_gb=4.0, reuse_from=None)
        staged.setup(self.source, self.samples_path, self.args)
        text = staged.function_chunks(self.sample)[0]
        self.coarse = {"source_index": 1, "rate": 1.0, "original_background_tokens": len(text),
                       "selected_functions": [0], "demonstrations_sort": [[0, 1.0]], "seconds": 1.0}
        chunker = EntropyChunking.__new__(EntropyChunking)
        sentences = chunker.split_into_sentences(text)
        chunks, _, ppls, spikes = chunker.chunk_from_ppls(sentences, [1., 5.] * 6 + [1.])
        self.entropy = {"source_index": 1, "seconds": 2.0, "functions": {"0": {
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "sentences": sentences,
            "ppls": ppls, "chunks": chunks, "spike_indices": spikes, "chunking_mode": "source"}}}
        _, allocations = allocate_fine_grained_ratios(
            [{"index": 0, "line_count": len(text.split("\n")), "token_count": len(text), "score": 1.0}],
            0.5, 5, 0.0, "token")
        self.result = {"source_index": 1, "id": "fixture", "gt": "return 1", "output": "return 1",
                       "prompt": text + "\n\ndef g():", "prompt_tokens": len(text) + 10,
                       "original_background_tokens": len(text), "compressed_background_tokens": len(text),
                       "compression_factor": 1.0, "fine_budget_mode": "token", "chunking_mode": "source",
                       "fine_budget_allocations": {str(i): row for i, row in allocations.items()},
                       "es": 100.0, "em": 100, "seconds": 3.0}
        for stage, row in (("coarse", self.coarse), ("entropy", self.entropy), ("results", self.result)):
            staged.atomic_json(self.source / stage / "1.json", row)
        staged.atomic_json(self.source / "summary.json", staged.summarize_rows([self.result]))
        self.args.chunking_mode = "paper"
        self.args.reuse_from = self.source

    def prepare(self):
        staged.setup(self.paper, self.samples_path, self.args)
        staged.stage_prepare(self.paper, self.samples, self.args)

    def finish(self):
        # Use the real LongCodeZip.compress_code_file, replacing only model I/O.
        engine = LongCodeZip.__new__(LongCodeZip)
        engine.get_token_length = lambda text, **kwargs: len(text)
        engine.get_ppl = lambda text, **kwargs: {"ppl": 1.0 if kwargs.get("condition_mode") == "prefix" else 2.0}
        engine.tokenizer = SimpleNamespace(encode=lambda text: list(text))
        engine.engine = SimpleNamespace(generate=Mock(return_value=[
            SimpleNamespace(outputs=[SimpleNamespace(text="return 1")])]))
        constructor = Mock(return_value=engine)
        with patch.dict(sys.modules, {
            "demo_awq": SimpleNamespace(VllmCompressor=constructor),
            "vllm": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
        }):
            staged.stage_fine(self.paper, self.samples, self.args)
        constructor.assert_called_once()
        return engine

    def test_prepare_uses_no_model_preserves_source_and_is_idempotent(self):
        before = {str(path.relative_to(self.source)): path.read_bytes() for path in self.source.rglob("*.json")}
        with patch.object(EntropyChunking, "__init__", side_effect=AssertionError("loaded a model")):
            self.prepare()
            first = {str(path.relative_to(self.paper)): path.read_bytes() for path in self.paper.rglob("*.json")}
            self.prepare()
        self.assertEqual(first, {str(path.relative_to(self.paper)): path.read_bytes() for path in self.paper.rglob("*.json")})
        self.assertEqual(before, {str(path.relative_to(self.source)): path.read_bytes() for path in self.source.rglob("*.json")})
        entropy = staged.read_json(self.paper / "entropy/1.json")
        self.assertEqual(entropy["functions"]["0"]["ppls"], self.entropy["functions"]["0"]["ppls"])
        self.assertNotEqual(entropy["functions"]["0"]["chunks"], self.entropy["functions"]["0"]["chunks"])
        self.assertEqual(entropy["timing_kind"], "cached_ppl_rechunking")

    def test_paper_mode_reaches_real_compressor_and_resume_needs_no_model(self):
        self.prepare()
        engine = self.finish()
        row = staged.read_json(self.paper / "results/1.json")
        self.assertEqual(row["chunking_mode"], "paper")
        self.assertEqual(row["fine_budget_allocations"], self.result["fine_budget_allocations"])
        self.assertIn("0", row["function_compressions"])
        self.assertIn("selected_blocks", row["function_compressions"]["0"])
        engine.engine.generate.assert_called_once()
        with patch.dict(sys.modules, {"demo_awq": SimpleNamespace(VllmCompressor=Mock(side_effect=AssertionError("loaded a model")))}):
            staged.stage_fine(self.paper, self.samples, self.args)
        metrics, counts = comparison.compare(self.source, self.paper, self.samples)
        self.assertEqual(metrics["average_es"], (100.0, 100.0))
        self.assertEqual(counts["matched_samples"], 1)

    def test_wrong_mode_is_rejected_before_loading_model(self):
        self.prepare()
        path = self.paper / "entropy/1.json"
        row = staged.read_json(path)
        row["functions"]["0"]["chunking_mode"] = "source"
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "text or mode"):
            staged.stage_fine(self.paper, self.samples, self.args)

    def test_default_source_mode_still_compresses_with_source_cache(self):
        self.prepare()
        engine = self.finish()
        engine.entropy_chunking = staged.CachedEntropy(self.entropy["functions"], "source")
        result = engine.compress_code_file(
            code=self.sample["background_context"], query=self.sample["current_function_context"],
            fine_ratio=0.5, fine_budget_mode="token", importance_beta=0.0)
        self.assertEqual(result["chunking_mode"], "source")
        self.assertEqual(result["fine_budget_allocations"][0], self.result["fine_budget_allocations"]["0"])

    def test_missing_selected_function_is_rejected_before_loading_model(self):
        self.prepare()
        path = self.paper / "entropy/1.json"
        row = staged.read_json(path)
        row["functions"].clear()
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "selected functions"):
            staged.stage_fine(self.paper, self.samples, self.args)

    def test_stale_result_is_rejected_before_loading_model(self):
        self.prepare()
        self.finish()
        path = self.paper / "results/1.json"
        row = staged.read_json(path)
        row["entropy_sha256"] = "stale"
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "stale entropy_sha256"):
            staged.stage_fine(self.paper, self.samples, self.args)

    def test_changed_shared_scores_are_rejected_before_loading_model(self):
        self.prepare()
        path = self.paper / "entropy/1.json"
        row = staged.read_json(path)
        cached = row["functions"]["0"]
        cached["ppls"][1] = 6.0
        chunker = EntropyChunking.__new__(EntropyChunking)
        chunks, _, _, spikes = chunker.chunk_from_ppls(cached["sentences"], cached["ppls"], chunking_mode="paper")
        cached.update(chunks=chunks, spike_indices=spikes)
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "Shared per-line inputs differ"):
            staged.stage_fine(self.paper, self.samples, self.args)

    def test_changed_reference_is_rejected_on_resume(self):
        self.prepare()
        row = copy.deepcopy(self.coarse)
        row["demonstrations_sort"][0][1] += 0.1
        staged.atomic_json(self.source / "coarse/1.json", row)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            staged.setup(self.paper, self.samples_path, self.args)

    def test_incomplete_reference_and_duplicate_samples_are_rejected(self):
        (self.source / "results/1.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            self.prepare()
        self.samples_path.write_bytes(self.samples_path.read_bytes() * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()

    def test_comparison_rejects_missing_results(self):
        self.prepare()
        self.finish()
        (self.paper / "results/1.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            comparison.compare(self.source, self.paper, self.samples)

    def test_comparison_rejects_budget_changes(self):
        self.prepare()
        self.finish()
        path = self.paper / "results/1.json"
        row = staged.read_json(path)
        row["fine_budget_allocations"]["0"]["target_tokens"] += 1
        staged.atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "budgets"):
            comparison.compare(self.source, self.paper, self.samples)

    def test_comparison_rejects_coarse_changes_even_when_local_hashes_match(self):
        self.prepare()
        self.finish()
        coarse_path = self.paper / "coarse/1.json"
        coarse = staged.read_json(coarse_path)
        coarse["demonstrations_sort"][0][1] += 0.1
        staged.atomic_json(coarse_path, coarse)
        entropy_path = self.paper / "entropy/1.json"
        entropy = staged.read_json(entropy_path)
        entropy["coarse_sha256"] = staged.coarse_fingerprint(coarse)
        staged.atomic_json(entropy_path, entropy)
        result_path = self.paper / "results/1.json"
        result = staged.read_json(result_path)
        result.update(coarse_sha256=staged.coarse_fingerprint(coarse), entropy_sha256=staged.entropy_fingerprint(entropy))
        staged.atomic_json(result_path, result)
        with self.assertRaisesRegex(ValueError, "Coarse selections or scores"):
            comparison.compare(self.source, self.paper, self.samples)

    def migrate(self):
        args = copy.copy(self.args)
        args.chunking_mode = "source"
        args.reuse_from = None
        args.migrate_from = self.source
        staged.setup(self.paper, self.samples_path, args)
        staged.stage_migrate(self.paper, self.samples, args)
        return args

    def test_legacy_migration_preserves_results_and_resumes_without_models(self):
        before = {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob("*.json")}
        args = self.migrate()
        row = staged.read_json(self.paper / "results/1.json")
        self.assertEqual(row["output"], self.result["output"])
        self.assertIn("entropy_sha256", row)
        with patch.dict(sys.modules, {"demo_awq": SimpleNamespace(VllmCompressor=Mock(side_effect=AssertionError("loaded a model")))}):
            staged.stage_coarse(self.paper, self.samples, args)
            staged.stage_entropy(self.paper, self.samples, args)
            staged.stage_fine(self.paper, self.samples, args)
        self.migrate()
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob("*.json")})

    def test_partial_legacy_migration_imports_only_complete_dependencies(self):
        (self.source / "results/1.json").unlink()
        (self.source / "entropy/1.json").unlink()
        self.migrate()
        self.assertTrue((self.paper / "coarse/1.json").is_file())
        self.assertFalse((self.paper / "entropy/1.json").exists())
        self.assertFalse((self.paper / "summary.json").exists())

    def test_early_results_without_budget_records_are_not_invented(self):
        row = copy.deepcopy(self.result)
        del row["fine_budget_allocations"]
        staged.atomic_json(self.source / "results/1.json", row)
        self.migrate()
        self.assertTrue((self.paper / "entropy/1.json").is_file())
        self.assertFalse((self.paper / "results/1.json").exists())
        self.assertEqual(staged.read_json(self.source / "results/1.json"), row)

    def test_migration_rejects_orphaned_entropy_before_writing_checkpoints(self):
        (self.source / "coarse/1.json").unlink()
        with self.assertRaisesRegex(ValueError, "no coarse input"):
            self.migrate()
        self.assertFalse((self.paper / "entropy/1.json").exists())

    def test_migration_rejects_changed_source_and_conflicting_target(self):
        self.migrate()
        row = staged.read_json(self.paper / "coarse/1.json")
        row["rate"] = 0.9
        staged.atomic_json(self.paper / "coarse/1.json", row)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.migrate()
        staged.atomic_json(self.source / "coarse/1.json", row)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.migrate()

    def test_migration_cannot_change_budget_settings(self):
        self.args.fine_ratio = 0.6
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.migrate()

    def test_shell_driver_migrates_legacy_data_without_running_models(self):
        command = ["bash", str(EXPERIMENT_DIR / "run_staged_awq.sh"), "--migrate-only",
                   "--legacy-from", str(self.source), "--outdir", str(self.paper),
                   "--samples", str(self.samples_path), "--fine-budget-mode", "token",
                   "--max-model-len", "16384", "--gpu-memory-utilization", "0.84", "--cpu-offload-gb", "4"]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("no model inference started", completed.stdout)
        self.assertTrue((self.paper / "results/1.json").is_file())
        command.remove("--migrate-only")
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("All fine checkpoints validated; no model loaded", completed.stdout)
