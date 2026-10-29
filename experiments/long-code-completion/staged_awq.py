"""Staged GPU pilot. Run each phase in a separate process to release GPU memory."""
import argparse
import hashlib
import json
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


def setup(outdir, samples_path, args):
    samples_bytes = samples_path.read_bytes()
    samples = [json.loads(line) for line in samples_bytes.decode("utf-8").splitlines() if line.strip()]
    config = {"samples_sha256": hashlib.sha256(samples_bytes).hexdigest(), "model": MODEL,
              "target_tokens": args.target_tokens, "fine_ratio": args.fine_ratio,
              "importance_beta": 0.0, "max_model_len": args.max_model_len,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "cpu_offload_gb": args.cpu_offload_gb}
    manifest = outdir / "manifest.json"
    if manifest.exists():
        existing = read_json(manifest)
        existing.setdefault("cpu_offload_gb", 0)
        if existing != config:
            raise ValueError("Existing manifest differs from samples or settings; choose a new output directory")
    else:
        atomic_json(manifest, config)
    return samples


def stage_coarse(outdir, samples, args):
    from demo_awq import VllmCompressor
    engine = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization,
                            load_entropy=False, cpu_offload_gb=args.cpu_offload_gb)
    for sample in samples:
        idx = sample_id(sample)
        path = outdir / "coarse" / (idx + ".json")
        if path.exists():
            print(f"coarse {idx}: already complete", flush=True)
            continue
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


def stage_entropy(outdir, samples):
    import torch
    from longcodezip import EntropyChunking, LongCodeZip
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU entropy scoring")
    entropy = EntropyChunking()
    if entropy.device.type != "cuda":
        raise RuntimeError(f"Entropy model is not on GPU: {entropy.device}")
    splitter = LongCodeZip.__new__(LongCodeZip)
    for sample in samples:
        idx = sample_id(sample)
        path = outdir / "entropy" / (idx + ".json")
        if path.exists():
            print(f"entropy {idx}: already complete", flush=True)
            continue
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        background = sample["background_context"].replace("<|endoftext|>", "")
        chunks = splitter.split_code_by_functions(background, language="python")
        outputs = {}
        start = time.monotonic()
        for i in coarse["selected_functions"]:
            print(f"entropy {idx}: function {i}", flush=True)
            parts, sentences, ppls, spikes = entropy.chunk_text_adaptive(chunks[i], method="std", k=0.2)
            outputs[str(i)] = {"text_sha256": hashlib.sha256(chunks[i].encode()).hexdigest(),
                               "chunks": parts, "sentences": sentences, "ppls": ppls,
                               "spike_indices": [int(x) for x in spikes]}
        atomic_json(path, {"source_index": sample["source_index"], "functions": outputs,
                           "seconds": time.monotonic() - start})
        print(f"entropy {idx}: done", flush=True)


class CachedEntropy:
    def __init__(self, functions):
        self.by_hash = {row["text_sha256"]: row for row in functions.values()}

    def chunk_text_adaptive(self, text, method="std", k=0.2):
        if method != "std" or k != 0.2:
            raise ValueError("Unexpected entropy settings")
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.by_hash:
            raise ValueError("Missing or changed function in entropy checkpoint")
        row = self.by_hash[key]
        return row["chunks"], row["sentences"], row["ppls"], row["spike_indices"]


def stage_fine(outdir, samples, args):
    from demo_awq import VllmCompressor
    from vllm import SamplingParams
    engine = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization,
                            load_entropy=False, cpu_offload_gb=args.cpu_offload_gb)
    params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
    for sample in samples:
        idx = sample_id(sample)
        path = outdir / "results" / (idx + ".json")
        if path.exists():
            print(f"fine {idx}: already complete", flush=True)
            continue
        coarse = read_json(outdir / "coarse" / (idx + ".json"))
        entropy = read_json(outdir / "entropy" / (idx + ".json"))
        engine.entropy_chunking = CachedEntropy(entropy["functions"])
        selected = coarse["selected_functions"]
        ranking = coarse["demonstrations_sort"]
        engine.control_context_budget = lambda contexts, **kw: ([contexts[i] for i in selected], selected, [0.0] * len(selected), ranking)
        start = time.monotonic()
        background = sample["background_context"].replace("<|endoftext|>", "")
        current = sample["current_function_context"]
        print(f"fine {idx}: scoring blocks and generating", flush=True)
        result = engine.compress_code_file(code=background, query=current, instruction=INSTRUCTION,
            rate=coarse["rate"], language="python", rank_only=False,
            fine_ratio=args.fine_ratio, importance_beta=0.0)
        compressed_code = result["compressed_code"]
        after = len(engine.tokenizer.encode(compressed_code))
        prompt = (compressed_code + "\n\n" + current).strip()
        prompt_tokens = len(engine.tokenizer.encode(prompt))
        if prompt_tokens + 128 > args.max_model_len:
            raise ValueError(f"Completion prompt has {prompt_tokens} tokens; limit {args.max_model_len - 128}")
        output = engine.engine.generate([prompt], params, use_tqdm=False)[0].outputs[0].text
        atomic_json(path, {"source_index": sample["source_index"], "id": sample["id"], "gt": sample["gt"],
            "original_background_tokens": coarse["original_background_tokens"],
            "compressed_background_tokens": after, "prompt_tokens": prompt_tokens,
            "compression_factor": coarse["original_background_tokens"] / after,
            "prompt": prompt, "output": output, "es": compute_ES(sample["gt"], output),
            "em": compute_EM(sample["gt"], output), "seconds": time.monotonic() - start})
        print(f"fine {idx}: done", flush=True)
    rows = [read_json(outdir / "results" / (sample_id(s) + ".json")) for s in samples]
    summary = {"count": len(rows), "average_es": sum(r["es"] for r in rows) / len(rows),
               "average_em": sum(r["em"] for r in rows) / len(rows),
               "aggregate_background_compression_factor": sum(r["original_background_tokens"] for r in rows) / sum(r["compressed_background_tokens"] for r in rows),
               "mean_sample_compression_factor": sum(r["compression_factor"] for r in rows) / len(rows)}
    atomic_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["coarse", "entropy", "fine"])
    p.add_argument("--samples", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-pilot/samples.jsonl")
    p.add_argument("--outdir", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-staged")
    p.add_argument("--target-tokens", type=int, default=2048)
    p.add_argument("--fine-ratio", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    p.add_argument("--cpu-offload-gb", type=float, default=0)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    samples = setup(args.outdir, args.samples, args)
    if not samples:
        raise ValueError("Sample list is empty")
    if args.stage == "coarse":
        stage_coarse(args.outdir, samples, args)
    elif args.stage == "entropy":
        stage_entropy(args.outdir, samples)
    else:
        stage_fine(args.outdir, samples, args)
