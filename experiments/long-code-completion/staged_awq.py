"""Staged GPU pilot. Run each phase in a separate process to release GPU memory."""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pilot_awq import INSTRUCTION, MODEL
from utils import compute_EM, compute_ES


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".staged-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sample_id(sample):
    return str(sample["source_index"])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def coarse_fingerprint(row):
    return fingerprint({key: row[key] for key in (
        "source_index", "rate", "original_background_tokens", "selected_functions", "demonstrations_sort"
    )})


def entropy_fingerprint(row):
    return fingerprint({key: row.get(key) for key in ("source_index", "functions", "coarse_sha256")})


def normalized_config(config):
    config = dict(config)
    config.setdefault("cpu_offload_gb", 0)
    config.setdefault("fine_budget_mode", "line")
    config.setdefault("chunking_mode", "source")
    config.setdefault("block_value_mode", "raw")
    return config


def common_config(config):
    return {key: value for key, value in normalized_config(config).items()
            if key not in {"chunking_mode", "shared_reference"}}


def shared_input_config(config):
    # Normalization and the AMI cache location do not affect coarse/entropy inputs.
    return {key: value for key, value in common_config(config).items()
            if key not in {"block_value_mode", "ami_cache"}}


def checkpoint_paths(directory, stage, samples):
    paths = {path.stem: path for path in (directory / stage).glob("*.json")}
    expected = {sample_id(sample) for sample in samples}
    if set(paths) != expected:
        raise ValueError(f"{directory / stage}: missing {len(expected - set(paths))}, "
                         f"unexpected {len(set(paths) - expected)} checkpoints")
    return paths


def function_chunks(sample):
    from longcodezip import LongCodeZip
    background = sample["background_context"].replace("<|endoftext|>", "")
    return LongCodeZip.__new__(LongCodeZip).split_code_by_functions(background, language="python")


def validate_coarse(sample, row):
    chunks = function_chunks(sample)
    selected = row["selected_functions"]
    ranked = [item[0] for item in row["demonstrations_sort"]]
    if (row["source_index"] != sample["source_index"] or
            len(selected) != len(set(selected)) or len(ranked) != len(set(ranked)) or
            not set(selected) <= set(ranked) or
            any(not isinstance(i, int) or i < 0 or i >= len(chunks) for i in selected + ranked) or
            row["original_background_tokens"] <= 0 or not 0 < row["rate"] <= 1):
        raise ValueError(f"Invalid coarse checkpoint for sample {sample_id(sample)}")
    return chunks


def validate_entropy(sample, coarse, row, mode, allow_legacy=False):
    from longcodezip import EntropyChunking
    chunks = validate_coarse(sample, coarse)
    expected_hash = coarse_fingerprint(coarse)
    if (row["source_index"] != sample["source_index"] or
            set(row["functions"]) != {str(i) for i in coarse["selected_functions"]} or
            row.get("coarse_sha256", expected_hash if allow_legacy else None) != expected_hash):
        raise ValueError(f"Entropy checkpoint has different selected functions or coarse input: {sample_id(sample)}")
    chunker = EntropyChunking.__new__(EntropyChunking)
    for index in coarse["selected_functions"]:
        cached = row["functions"][str(index)]
        text = chunks[index]
        if (cached["text_sha256"] != hashlib.sha256(text.encode()).hexdigest() or
                cached.get("chunking_mode", "source") != mode or
                cached["sentences"] != chunker.split_into_sentences(text)):
            raise ValueError(f"Entropy checkpoint text or mode differs: {sample_id(sample)}/{index}")
        rebuilt = chunker.chunk_from_ppls(cached["sentences"], cached["ppls"], chunking_mode=mode)
        if rebuilt[0] != cached["chunks"] or rebuilt[3] != cached["spike_indices"]:
            raise ValueError(f"Entropy checkpoint boundaries differ: {sample_id(sample)}/{index}")


def validate_result(sample, coarse, entropy, row, config, allow_legacy=False):
    if (row["source_index"] != sample["source_index"] or row["id"] != sample["id"] or
            row["gt"] != sample["gt"] or
            row.get("chunking_mode", "source") != config["chunking_mode"] or
            row.get("fine_budget_mode", "line") != config["fine_budget_mode"] or
            row.get("block_value_mode", "raw") != config.get("block_value_mode", "raw") or
            row["original_background_tokens"] != coarse["original_background_tokens"]):
        raise ValueError(f"Result settings differ for sample {sample_id(sample)}")
    for key, expected in (("coarse_sha256", coarse_fingerprint(coarse)),
                          ("entropy_sha256", entropy_fingerprint(entropy))):
        if row.get(key, expected if allow_legacy else None) != expected:
            raise ValueError(f"Result has missing or stale {key}: {sample_id(sample)}; choose a new output directory")
    if set(row["fine_budget_allocations"]) != {str(i) for i in coarse["selected_functions"]}:
        raise ValueError(f"Result uses different function budgets: {sample_id(sample)}")
    from longcodezip.budget import allocate_fine_grained_ratios
    chunks = function_chunks(sample)
    scores = dict(coarse["demonstrations_sort"])
    functions = []
    for index in coarse["selected_functions"]:
        allocation = row["fine_budget_allocations"][str(index)]
        functions.append({"index": index, "line_count": len(chunks[index].split("\n")),
                          "token_count": allocation["original_tokens"], "score": scores[index]})
    _, expected_allocations = allocate_fine_grained_ratios(
        functions, config["fine_ratio"], 5, config["importance_beta"], config["fine_budget_mode"])
    if row["fine_budget_allocations"] != {str(i): value for i, value in expected_allocations.items()}:
        raise ValueError(f"Result budgets do not match the allocation settings: {sample_id(sample)}")
    if (not math.isclose(row["es"], compute_ES(sample["gt"], row["output"])) or
            row["em"] != compute_EM(sample["gt"], row["output"]) or
            row["compressed_background_tokens"] <= 0 or
            not math.isclose(row["compression_factor"],
                             row["original_background_tokens"] / row["compressed_background_tokens"])):
        raise ValueError(f"Result metrics differ for sample {sample_id(sample)}")


def validate_shared_inputs(sample, coarse, entropy, reference):
    name = sample_id(sample) + ".json"
    original_coarse = read_json(reference / "coarse" / name)
    original_entropy = read_json(reference / "entropy" / name)
    if coarse_fingerprint(coarse) != coarse_fingerprint(original_coarse):
        raise ValueError(f"Shared coarse inputs differ for sample {sample_id(sample)}")
    for index, original in original_entropy["functions"].items():
        cached = entropy["functions"].get(index)
        if cached is None or any(fingerprint(cached[key]) != fingerprint(original[key])
                                 for key in ("text_sha256", "sentences", "ppls")):
            raise ValueError(f"Shared per-line inputs differ for sample {sample_id(sample)}/{index}")


def summarize_rows(rows):
    return {"count": len(rows), "average_es": sum(r["es"] for r in rows) / len(rows),
            "average_em": sum(r["em"] for r in rows) / len(rows),
            "aggregate_background_compression_factor": sum(r["original_background_tokens"] for r in rows) / sum(r["compressed_background_tokens"] for r in rows),
            "mean_sample_compression_factor": sum(r["compression_factor"] for r in rows) / len(rows)}


def setup(outdir, samples_path, args):
    samples_bytes = samples_path.read_bytes()
    samples = [json.loads(line) for line in samples_bytes.decode("utf-8").splitlines() if line.strip()]
    if not samples or len({sample_id(sample) for sample in samples}) != len(samples):
        raise ValueError("Sample list is empty or contains duplicate source indices")
    config = {"samples_sha256": hashlib.sha256(samples_bytes).hexdigest(), "model": MODEL,
              "target_tokens": args.target_tokens, "fine_ratio": args.fine_ratio,
              "fine_budget_mode": args.fine_budget_mode,
              "chunking_mode": args.chunking_mode,
              "block_value_mode": getattr(args, "block_value_mode", "raw"),
              "importance_beta": 0.0, "max_model_len": args.max_model_len,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "cpu_offload_gb": args.cpu_offload_gb}
    reference = getattr(args, "reuse_from", None)
    if reference is not None:
        reference = reference.resolve()
        if reference == outdir.resolve():
            raise ValueError("Output directory must differ from the shared reference")
        reference_config = normalized_config(read_json(reference / "manifest.json"))
        if reference_config["chunking_mode"] != "source" or shared_input_config(reference_config) != shared_input_config(config):
            raise ValueError("Shared source reference differs from samples or settings")
        hashes = {}
        for stage in ("coarse", "entropy", "results"):
            paths = checkpoint_paths(reference, stage, samples)
            hashes[stage] = fingerprint({idx: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for idx, path in paths.items()})
        config["shared_reference"] = {"directory": str(reference), "checkpoint_sha256": hashes,
                                      "manifest_sha256": hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest()}
    ami_cache = getattr(args, "ami_cache", None)
    if ami_cache is not None:
        ami_cache = ami_cache.resolve()
        cache_config = normalized_config(read_json(ami_cache / "manifest.json"))
        if (shared_input_config(cache_config) != shared_input_config(config) or
                cache_config["chunking_mode"] != config["chunking_mode"] or
                cache_config["block_value_mode"] != "raw" or "ami_cache" in cache_config):
            raise ValueError("AMI cache experiment settings differ")
        paths = {path.stem: path for path in (ami_cache / "ami").glob("*.json")}
        if set(paths) - {sample_id(sample) for sample in samples}:
            raise ValueError("AMI cache contains unexpected sample IDs")
        config["ami_cache"] = {"directory": str(ami_cache),
                               "manifest_sha256": hashlib.sha256((ami_cache / "manifest.json").read_bytes()).hexdigest()}
    manifest = outdir / "manifest.json"
    if manifest.exists():
        existing = normalized_config(read_json(manifest))
        if existing != config:
            raise ValueError("Existing manifest differs from samples or settings; choose a new output directory")
    else:
        atomic_json(manifest, config)
    return samples


def stage_prepare(outdir, samples, args):
    """Rebuild boundary caches from a validated, completed source experiment."""
    from longcodezip import EntropyChunking
    if args.reuse_from is None:
        raise ValueError("prepare requires --reuse-from pointing to a completed source experiment")
    reference = args.reuse_from.resolve()
    reference_config = normalized_config(read_json(reference / "manifest.json"))
    chunker = EntropyChunking.__new__(EntropyChunking)
    prepared = []
    for sample in samples:
        name = sample_id(sample) + ".json"
        coarse = read_json(reference / "coarse" / name)
        entropy = read_json(reference / "entropy" / name)
        result = read_json(reference / "results" / name)
        validate_entropy(sample, coarse, entropy, "source", allow_legacy=True)
        validate_result(sample, coarse, entropy, result, reference_config, allow_legacy=True)
        prepared.append((sample, coarse, entropy))
    # Validate the whole reference before creating any reusable checkpoints.
    for sample, coarse, entropy in prepared:
        name = sample_id(sample) + ".json"
        start = time.monotonic()
        functions = {}
        for index, cached in entropy["functions"].items():
            chunks, sentences, ppls, spikes = chunker.chunk_from_ppls(
                cached["sentences"], cached["ppls"], chunking_mode=args.chunking_mode)
            functions[index] = {**cached, "chunks": chunks, "spike_indices": spikes,
                                "chunking_mode": args.chunking_mode}
        new_coarse = {**coarse, "seconds": 0.0, "timing_kind": "reused",
                      "reused_scoring_seconds": coarse["seconds"]}
        rebuilt = {"source_index": sample["source_index"], "functions": functions,
                   "coarse_sha256": coarse_fingerprint(coarse),
                   "seconds": time.monotonic() - start, "timing_kind": "cached_ppl_rechunking",
                   "reused_scoring_seconds": entropy["seconds"]}
        for stage, row, digest in (("coarse", new_coarse, coarse_fingerprint),
                                   ("entropy", rebuilt, entropy_fingerprint)):
            path = outdir / stage / name
            if path.exists():
                if digest(read_json(path)) != digest(row):
                    raise ValueError(f"Existing {stage} cache differs: {path}; choose a new directory")
            else:
                atomic_json(path, row)
    print(f"Prepared {len(samples)} paired samples using cached per-line scores; no model loaded", flush=True)


def stage_migrate(outdir, samples, args):
    """Import verifiable legacy checkpoints into a separate output directory."""
    source = args.migrate_from.resolve()
    if source == outdir.resolve():
        raise ValueError("Migration must use a new output directory")
    config = normalized_config(read_json(outdir / "manifest.json"))
    old_config = normalized_config(read_json(source / "manifest.json"))
    if config != old_config:
        raise ValueError("Legacy experiment settings differ; migration cannot change the experiment")
    expected = {sample_id(sample) for sample in samples}
    paths = {stage: {p.stem: p for p in (source / stage).glob("*.json")}
             for stage in ("coarse", "entropy", "results")}
    for stage, files in paths.items():
        if set(files) - expected:
            raise ValueError(f"Unexpected legacy samples in {source / stage}")
    provenance = {"directory": str(source), "manifest_sha256": fingerprint(old_config),
                  "checkpoint_sha256": {stage: fingerprint({idx: hashlib.sha256(p.read_bytes()).hexdigest()
                                                            for idx, p in files.items()})
                                        for stage, files in paths.items()}}
    record = outdir / "migration.json"
    if record.exists() and read_json(record) != provenance:
        raise ValueError("Legacy source changed after migration; choose a new output directory")
    prepared = []
    skipped = 0
    for sample in samples:
        idx = sample_id(sample)
        coarse = read_json(paths["coarse"][idx]) if idx in paths["coarse"] else None
        entropy = read_json(paths["entropy"][idx]) if idx in paths["entropy"] else None
        result = read_json(paths["results"][idx]) if idx in paths["results"] else None
        if coarse is not None:
            validate_coarse(sample, coarse)
        if entropy is not None:
            if coarse is None:
                raise ValueError(f"Legacy entropy has no coarse input: {idx}")
            validate_entropy(sample, coarse, entropy, config["chunking_mode"], allow_legacy=True)
        if result is not None:
            if entropy is None:
                raise ValueError(f"Legacy result has no entropy input: {idx}")
            if "fine_budget_allocations" not in result:
                # Preserve the original result on disk; never invent missing budget records.
                if (result["source_index"] != sample["source_index"] or result["id"] != sample["id"] or
                        result["gt"] != sample["gt"] or
                        not math.isclose(result["es"], compute_ES(sample["gt"], result["output"])) or
                        result["em"] != compute_EM(sample["gt"], result["output"])):
                    raise ValueError(f"Invalid legacy result: {idx}")
                result = None
                skipped += 1
            else:
                validate_result(sample, coarse, entropy, result, config, allow_legacy=True)
        prepared.append((sample, coarse, entropy, result))
    counts = {stage: 0 for stage in paths}
    for sample, coarse, entropy, result in prepared:
        name = sample_id(sample) + ".json"
        if coarse is None:
            continue
        coarse = {**coarse, "timing_kind": "reused", "legacy_source": str(source / "coarse" / name)}
        if entropy is not None:
            entropy = {**entropy, "coarse_sha256": coarse_fingerprint(coarse),
                       "timing_kind": "reused", "legacy_source": str(source / "entropy" / name)}
        if result is not None:
            result = {**result, "coarse_sha256": coarse_fingerprint(coarse),
                      "entropy_sha256": entropy_fingerprint(entropy), "timing_kind": "reused",
                      "legacy_source": str(source / "results" / name)}
        for stage, row, digest in (("coarse", coarse, coarse_fingerprint),
                                   ("entropy", entropy, entropy_fingerprint), ("results", result, None)):
            if row is None:
                continue
            path = outdir / stage / name
            if path.exists():
                existing = read_json(path)
                if stage == "results":
                    validate_result(sample, coarse, entropy, existing, config)
                elif digest(existing) != digest(row):
                    raise ValueError(f"Migration conflicts with existing {path}")
            else:
                atomic_json(path, row)
            counts[stage] += 1
    atomic_json(record, provenance)
    if len(list((outdir / "results").glob("*.json"))) == len(samples):
        rows = [read_json(outdir / "results" / (sample_id(s) + ".json")) for s in samples]
        atomic_json(outdir / "summary.json", summarize_rows(rows))
    print(f"Migrated checkpoints: {counts}; legacy results requiring recomputation: {skipped}; no model loaded", flush=True)


def stage_coarse(outdir, samples, args):
    pending = []
    for sample in samples:
        path = outdir / "coarse" / (sample_id(sample) + ".json")
        if path.exists():
            validate_coarse(sample, read_json(path))
        else:
            pending.append(sample)
    if not pending:
        print("All coarse checkpoints validated; no model loaded", flush=True)
        return
    from demo_awq import VllmCompressor
    engine = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization,
                            load_entropy=False, cpu_offload_gb=args.cpu_offload_gb)
    for sample in pending:
        idx = sample_id(sample)
        path = outdir / "coarse" / (idx + ".json")
        start = time.monotonic()
        background = sample["background_context"].replace("<|endoftext|>", "")
        original_tokens = len(engine.tokenizer.encode(background))
        rate = min(1.0, args.target_tokens / original_tokens)
        print(f"coarse {idx}: scoring functions", flush=True)
        result = engine.compress_code_file(code=background, query=sample["current_function_context"],
            instruction=INSTRUCTION, rate=rate, language="python", rank_only=True)
        atomic_json(path, {"source_index": sample["source_index"], "rate": rate,
                           "original_background_tokens": original_tokens,
                           "selected_functions": result["selected_functions"],
                           "demonstrations_sort": result["demonstrations_sort"],
                           "seconds": time.monotonic() - start})
        print(f"coarse {idx}: done", flush=True)


def stage_entropy(outdir, samples, args):
    pending = []
    for sample in samples:
        idx = sample_id(sample)
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        validate_coarse(sample, coarse)
        path = outdir / "entropy" / (idx + ".json")
        if path.exists():
            validate_entropy(sample, coarse, read_json(path), args.chunking_mode)
        else:
            pending.append(sample)
    if not pending:
        print("All entropy checkpoints validated; no model loaded", flush=True)
        return
    import torch
    from longcodezip import EntropyChunking, LongCodeZip
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU entropy scoring")
    entropy = EntropyChunking()
    if entropy.device.type != "cuda":
        raise RuntimeError(f"Entropy model is not on GPU: {entropy.device}")
    splitter = LongCodeZip.__new__(LongCodeZip)
    for sample in pending:
        idx = sample_id(sample)
        path = outdir / "entropy" / (idx + ".json")
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        background = sample["background_context"].replace("<|endoftext|>", "")
        chunks = splitter.split_code_by_functions(background, language="python")
        outputs = {}
        start = time.monotonic()
        for i in coarse["selected_functions"]:
            print(f"entropy {idx}: function {i}", flush=True)
            parts, sentences, ppls, spikes = entropy.chunk_text_adaptive(
                chunks[i], method="std", k=0.2, chunking_mode=args.chunking_mode
            )
            outputs[str(i)] = {"text_sha256": hashlib.sha256(chunks[i].encode()).hexdigest(),
                               "chunks": parts, "sentences": sentences, "ppls": ppls,
                               "spike_indices": [int(x) for x in spikes],
                               "chunking_mode": args.chunking_mode}
        atomic_json(path, {"source_index": sample["source_index"], "functions": outputs,
                           "coarse_sha256": coarse_fingerprint(coarse),
                           "seconds": time.monotonic() - start, "timing_kind": "model_scoring"})
        print(f"entropy {idx}: done", flush=True)


class CachedEntropy:
    def __init__(self, functions, chunking_mode):
        if any(row.get("chunking_mode", "source") != chunking_mode for row in functions.values()):
            raise ValueError("Entropy checkpoint contains a different chunking mode")
        self.by_hash = {row["text_sha256"]: row for row in functions.values()}
        self.chunking_mode = chunking_mode

    def chunk_text_adaptive(self, text, method="std", k=0.2, chunking_mode="source"):
        if method != "std" or k != 0.2:
            raise ValueError("Unexpected entropy settings")
        if chunking_mode != self.chunking_mode:
            raise ValueError("Entropy cache uses a different chunking mode")
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.by_hash:
            raise ValueError("Missing or changed function in entropy checkpoint")
        row = self.by_hash[key]
        return row["chunks"], row["sentences"], row["ppls"], row["spike_indices"]


def working_samples(samples, args):
    limit = getattr(args, "max_samples", None)
    return samples if limit is None else samples[:limit]


def load_ami(outdir, sample, coarse, entropy, args):
    from ami_cache import validate_checkpoint
    row = read_json(outdir / "ami" / (sample_id(sample) + ".json"))
    reference = read_json(args.reuse_from / "results" / (sample_id(sample) + ".json"))
    validate_checkpoint(row, sample, entropy, reference["fine_budget_allocations"],
                        coarse_fingerprint(coarse), entropy_fingerprint(entropy))
    return row


def stage_ami(outdir, samples, args):
    """Calculate raw block scores once; selection/generation happens in fine."""
    if args.reuse_from is None or getattr(args, "ami_cache", None) is not None or getattr(args, "block_value_mode", "raw") != "raw":
        raise ValueError("ami requires --reuse-from, raw mode, and no --ami-cache")
    pending = []
    for sample in working_samples(samples, args):
        idx = sample_id(sample)
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        entropy = read_json(outdir / "entropy" / (idx + ".json"))
        validate_entropy(sample, coarse, entropy, args.chunking_mode)
        validate_shared_inputs(sample, coarse, entropy, args.reuse_from)
        if (outdir / "ami" / (idx + ".json")).exists():
            load_ami(outdir, sample, coarse, entropy, args)
        else:
            pending.append(sample)
    if not pending:
        print("Requested AMI checkpoints validated; no model loaded", flush=True)
        return
    from demo_awq import VllmCompressor
    from ami_cache import digest, query_digest, validate_checkpoint
    engine = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization,
                            load_entropy=False, cpu_offload_gb=args.cpu_offload_gb)
    for sample in pending:
        idx = sample_id(sample)
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        entropy = read_json(outdir / "entropy" / (idx + ".json"))
        engine.entropy_chunking = CachedEntropy(entropy["functions"], args.chunking_mode)
        selected, ranking = coarse["selected_functions"], coarse["demonstrations_sort"]
        engine.control_context_budget = lambda contexts, **kw: ([contexts[i] for i in selected], selected, [0.0] * len(selected), ranking)
        start = time.monotonic()
        print(f"ami {idx}: scoring blocks", flush=True)
        result = engine.compress_code_file(
            code=sample["background_context"].replace("<|endoftext|>", ""),
            query=sample["current_function_context"], instruction=INSTRUCTION,
            rate=coarse["rate"], language="python", fine_ratio=args.fine_ratio,
            importance_beta=0.0, fine_budget_mode=args.fine_budget_mode,
            chunking_mode=args.chunking_mode, block_value_mode="raw", score_only=True)
        functions = {str(i): {**value,
            "function_sha256": entropy["functions"][str(i)]["text_sha256"],
            "blocks_sha256": digest(entropy["functions"][str(i)]["chunks"])}
            for i, value in result["function_compressions"].items()}
        reference = read_json(args.reuse_from / "results" / (idx + ".json"))
        row = {"schema_version": 1, "source_index": sample["source_index"], "functions": functions,
               "query_sha256": query_digest(sample["current_function_context"]), "method": "conditional_ppl",
               "coarse_sha256": coarse_fingerprint(coarse), "entropy_sha256": entropy_fingerprint(entropy),
               "fine_budget_allocations": {str(i): value for i, value in result["fine_budget_allocations"].items()},
               "seconds": time.monotonic() - start}
        validate_checkpoint(row, sample, entropy, reference["fine_budget_allocations"],
                            coarse_fingerprint(coarse), entropy_fingerprint(entropy))
        atomic_json(outdir / "ami" / (idx + ".json"), row)
        print(f"ami {idx}: done", flush=True)


def stage_fine(outdir, samples, args):
    config = normalized_config(read_json(outdir / "manifest.json"))
    requested = working_samples(samples, args)
    ami_cache = getattr(args, "ami_cache", None)
    if ami_cache is not None and args.reuse_from is None:
        raise ValueError("--ami-cache requires --reuse-from")
    pending = []
    for sample in requested:
        idx = sample_id(sample)
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        entropy = read_json(outdir / "entropy" / (idx + ".json"))
        validate_entropy(sample, coarse, entropy, args.chunking_mode)
        if args.reuse_from is not None:
            validate_shared_inputs(sample, coarse, entropy, args.reuse_from)
        if ami_cache is not None:
            cache = load_ami(ami_cache, sample, coarse, entropy, args)
        path = outdir / "results" / (idx + ".json")
        if path.exists():
            row = read_json(path)
            validate_result(sample, coarse, entropy, row, config)
            if ami_cache is not None:
                from ami_cache import validate_result as validate_ami_result
                validate_ami_result(row, cache, entropy, config["block_value_mode"])
            if args.reuse_from is not None:
                reference_row = read_json(args.reuse_from / "results" / (idx + ".json"))
                if row["fine_budget_allocations"] != reference_row["fine_budget_allocations"]:
                    raise ValueError(f"Paired function budgets differ for sample {idx}")
        else:
            pending.append(sample)
    if not pending:
        if len(requested) != len(samples):
            print(f"Pilot results validated: {len(requested)}/{len(samples)}; no final summary or model loaded", flush=True)
            return
        rows = [read_json(outdir / "results" / (sample_id(s) + ".json")) for s in samples]
        atomic_json(outdir / "summary.json", summarize_rows(rows))
        print("All fine checkpoints validated; no model loaded", flush=True)
        return
    from demo_awq import VllmCompressor
    from vllm import SamplingParams
    engine = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization,
                            load_entropy=False, cpu_offload_gb=args.cpu_offload_gb)
    params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
    for sample in pending:
        idx = sample_id(sample)
        path = outdir / "results" / (idx + ".json")
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        entropy = read_json(outdir / "entropy" / (idx + ".json"))
        engine.entropy_chunking = CachedEntropy(entropy["functions"], args.chunking_mode)
        if ami_cache is not None:
            from ami_cache import CachedBlockScorer, digest
            cache = load_ami(ami_cache, sample, coarse, entropy, args)
            engine.score_blocks = CachedBlockScorer(cache).score_blocks
        selected = coarse["selected_functions"]
        ranking = coarse["demonstrations_sort"]
        engine.control_context_budget = lambda contexts, **kw: ([contexts[i] for i in selected], selected, [0.0] * len(selected), ranking)
        start = time.monotonic()
        background = sample["background_context"].replace("<|endoftext|>", "")
        current = sample["current_function_context"]
        print(f"fine {idx}: scoring blocks and generating", flush=True)
        result = engine.compress_code_file(code=background, query=current, instruction=INSTRUCTION,
            rate=coarse["rate"], language="python", rank_only=False,
            fine_ratio=args.fine_ratio, importance_beta=0.0,
            fine_budget_mode=args.fine_budget_mode, chunking_mode=args.chunking_mode,
            block_value_mode=getattr(args, "block_value_mode", "raw"))
        allocations = {str(i): value for i, value in result.get("fine_budget_allocations", {}).items()}
        if args.reuse_from is not None:
            reference_row = read_json(args.reuse_from / "results" / (idx + ".json"))
            if allocations != reference_row["fine_budget_allocations"]:
                raise ValueError(f"Paired function budgets differ for sample {idx}")
        if ami_cache is not None:
            from ami_cache import validate_result as validate_ami_result
            validate_ami_result({"ami_sha256": digest(cache),
                                 "block_value_mode": getattr(args, "block_value_mode", "raw"),
                                 "function_compressions": result["function_compressions"]},
                                cache, entropy, config["block_value_mode"])
        compressed_code = result["compressed_code"]
        after = len(engine.tokenizer.encode(compressed_code))
        prompt = (compressed_code + "\n\n" + current).strip()
        prompt_tokens = len(engine.tokenizer.encode(prompt))
        if prompt_tokens + 128 > args.max_model_len:
            raise ValueError(f"Completion prompt has {prompt_tokens} tokens; limit {args.max_model_len - 128}")
        compression_seconds = time.monotonic() - start
        generation_start = time.monotonic()
        output = engine.engine.generate([prompt], params, use_tqdm=False)[0].outputs[0].text
        row = {"source_index": sample["source_index"], "id": sample["id"], "gt": sample["gt"],
            "original_background_tokens": coarse["original_background_tokens"],
            "compressed_background_tokens": after, "prompt_tokens": prompt_tokens,
            "compression_factor": coarse["original_background_tokens"] / after,
            "fine_budget_mode": args.fine_budget_mode,
            "chunking_mode": args.chunking_mode,
            "block_value_mode": getattr(args, "block_value_mode", "raw"),
            "fine_budget_allocations": allocations,
            "function_compressions": result.get("function_compressions", {}),
            "coarse_sha256": coarse_fingerprint(coarse), "entropy_sha256": entropy_fingerprint(entropy),
            "compression_seconds": compression_seconds,
            "generation_seconds": time.monotonic() - generation_start,
            "prompt": prompt, "output": output, "es": compute_ES(sample["gt"], output),
            "em": compute_EM(sample["gt"], output), "seconds": time.monotonic() - start}
        if ami_cache is not None:
            row["ami_sha256"] = digest(cache)
            row["scoring_source"] = "shared_ami_cache"
            row["scoring_cache_read_seconds"] = sum(f["scoring_seconds"] for f in result["function_compressions"].values())
            row["shared_scoring_seconds"] = sum(f["scoring_seconds"] for f in cache["functions"].values())
            row["selection_seconds"] = sum(f["selection_seconds"] for f in result["function_compressions"].values())
            from ami_cache import validate_result as validate_ami_result
            validate_ami_result(row, cache, entropy, config["block_value_mode"])
        validate_result(sample, coarse, entropy, row, config)
        atomic_json(path, row)
        print(f"fine {idx}: done", flush=True)
    if len(requested) != len(samples):
        print(f"Pilot complete: {len(requested)}/{len(samples)}; full-run summary deferred", flush=True)
        return
    rows = [read_json(outdir / "results" / (sample_id(s) + ".json")) for s in samples]
    summary = summarize_rows(rows)
    atomic_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "migrate", "coarse", "entropy", "ami", "fine"])
    p.add_argument("--samples", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-pilot/samples.jsonl")
    p.add_argument("--outdir", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-staged")
    p.add_argument("--target-tokens", type=int, default=2048)
    p.add_argument("--fine-ratio", type=float, default=0.5)
    p.add_argument("--fine-budget-mode", choices=["line", "token"], default="line")
    p.add_argument("--chunking-mode", choices=["source", "paper"], default="source")
    p.add_argument("--block-value-mode", choices=["raw", "minmax"], default="raw")
    p.add_argument("--ami-cache", type=Path, help="Shared raw AMI checkpoint directory")
    p.add_argument("--max-samples", type=int, help="Process the first N samples for a resumable pilot")
    p.add_argument("--reuse-from", type=Path,
                   help="Completed source experiment supplying shared coarse decisions and per-line PPL")
    p.add_argument("--migrate-from", type=Path,
                   help="Legacy experiment to validate and import into a separate output directory")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    p.add_argument("--cpu-offload-gb", type=float, default=0)
    args = p.parse_args()
    if args.max_samples is not None and (args.max_samples <= 0 or args.stage not in {"ami", "fine"}):
        p.error("--max-samples must be positive and is only valid for ami or fine")
    if args.ami_cache is not None and (args.reuse_from is None or args.stage not in {"prepare", "fine"}):
        p.error("--ami-cache requires --reuse-from and the prepare or fine stage")
    if args.stage == "prepare" and args.reuse_from is None:
        p.error("prepare requires --reuse-from")
    if (args.stage == "migrate") != (args.migrate_from is not None):
        p.error("migrate requires --migrate-from, which is only valid for the migrate stage")
    if args.migrate_from is not None and args.reuse_from is not None:
        p.error("migration and paired preparation are separate operations")
    if args.reuse_from is not None and args.stage in {"coarse", "entropy"}:
        p.error("use prepare to import shared caches, then fine to finish the paired run")
    args.outdir.mkdir(parents=True, exist_ok=True)
    samples = setup(args.outdir, args.samples, args)
    if not samples:
        raise ValueError("Sample list is empty")
    if args.stage == "prepare":
        stage_prepare(args.outdir, samples, args)
    elif args.stage == "migrate":
        stage_migrate(args.outdir, samples, args)
    elif args.stage == "coarse":
        stage_coarse(args.outdir, samples, args)
    elif args.stage == "entropy":
        stage_entropy(args.outdir, samples, args)
    elif args.stage == "ami":
        stage_ami(args.outdir, samples, args)
    else:
        stage_fine(args.outdir, samples, args)
