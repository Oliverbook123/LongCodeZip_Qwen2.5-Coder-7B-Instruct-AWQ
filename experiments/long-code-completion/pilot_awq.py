"""Small, resumable LCC Python pilot using the repository's vLLM adapter."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import compute_EM, compute_ES, split_context_ast

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
INSTRUCTION = "Complete the following code function given the context."


def prepare(outdir, count, test_parquet):
    import requests
    from urllib3.util.retry import Retry
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    selected = []
    # Official 500-sample run examines the first 5000 rows before filtering.
    if test_parquet.is_file():
        dataset = load_dataset("parquet", data_files=str(test_parquet), split="train")
        rows = ((i, dataset[i]) for i in range(min(5000, len(dataset))))
    else:
        def remote_rows():
            session = requests.Session()
            retry = Retry(total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                          allowed_methods=["GET"], respect_retry_after_header=True)
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
            for offset in range(0, 5000, 100):
                page_path = outdir / "pages" / f"{offset:04d}.json"
                if page_path.exists():
                    page = json.loads(page_path.read_text(encoding="utf-8"))
                else:
                    print(f"Fetching test rows {offset}–{offset + 99}", flush=True)
                    response = session.get("https://datasets-server.huggingface.co/rows",
                        params={"dataset": "microsoft/LCC_python", "config": "default",
                                "split": "test", "offset": offset, "length": 100}, timeout=60)
                    response.raise_for_status()
                    page = response.json()
                    page_path.parent.mkdir(parents=True, exist_ok=True)
                    temp = page_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
                    os.replace(temp, page_path)
                for item in page["rows"]:
                    yield item["row_idx"], item["row"]
        rows = remote_rows()
    for index, row in rows:
        background, current = split_context_ast(row["context"])
        if len(current.splitlines()) > 50 or not background.strip():
            continue
        tokens = len(tokenizer.encode(background, add_special_tokens=False))
        if tokens < 5000:
            continue
        selected.append({"source_index": index, "id": row.get("id", index),
                         "background_context": background,
                         "current_function_context": current, "gt": row["gt"],
                         "filter_background_tokens": tokens})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} qualifying samples found in first 5000 rows")
    path = outdir / "samples.jsonl"
    with path.open("x", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(selected)} samples in {path}; indices: {[x['source_index'] for x in selected]}", flush=True)


def evaluate(outdir, args):
    import torch
    from vllm import SamplingParams
    from demo_awq import VllmCompressor

    torch.set_num_threads(args.cpu_threads)
    samples = [json.loads(line) for line in (outdir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    results_path = outdir / "results.jsonl"
    done = {}
    if results_path.exists():
        done = {r["source_index"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())}
    compressor = VllmCompressor(MODEL, args.max_model_len, args.gpu_memory_utilization)
    tokenizer = compressor.tokenizer
    params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
    for sample in samples:
        idx = sample["source_index"]
        if idx in done:
            continue
        start = time.monotonic()
        result = {"source_index": idx, "id": sample["id"], "gt": sample["gt"]}
        try:
            background = sample["background_context"].replace("<|endoftext|>", "")
            current = sample["current_function_context"]
            before = len(tokenizer.encode(background))
            rate = min(1.0, args.target_tokens / before)
            print(f"Starting sample {idx}: {before} background tokens", flush=True)
            compressed = compressor.compress_code_file(
                code=background, query=current, instruction=INSTRUCTION,
                rate=rate, language="python", rank_only=False,
                fine_ratio=args.fine_ratio, importance_beta=0.0)
            compressed_code = compressed["compressed_code"]
            after = len(tokenizer.encode(compressed_code))
            prompt = (compressed_code + "\n\n" + current).strip()
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens + 128 > args.max_model_len:
                raise ValueError(f"Completion prompt has {prompt_tokens} tokens, limit {args.max_model_len - 128}")
            output = compressor.engine.generate([prompt], params, use_tqdm=False)[0].outputs[0].text
            result.update(status="ok", original_background_tokens=before,
                          compressed_background_tokens=after,
                          compression_factor=before / after if after else None,
                          prompt_tokens=prompt_tokens, prompt=prompt, output=output,
                          es=compute_ES(sample["gt"], output), em=compute_EM(sample["gt"], output))
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        done[idx] = result
        successes = [x for x in done.values() if x["status"] == "ok"]
        summary = {"model": MODEL, "dataset": "microsoft/LCC_python/test",
                   "target_tokens": args.target_tokens, "fine_ratio": args.fine_ratio,
                   "importance_beta": 0.0, "max_model_len": args.max_model_len,
                   "total_selected": len(samples), "completed": len(done),
                   "successful": len(successes), "failed": len(done) - len(successes),
                   "average_es": sum(x["es"] for x in successes) / len(successes) if successes else None,
                   "average_em": sum(x["em"] for x in successes) / len(successes) if successes else None,
                   "mean_sample_compression_factor": sum(x["compression_factor"] for x in successes) / len(successes) if successes else None,
                   "aggregate_background_compression_factor": sum(x["original_background_tokens"] for x in successes) / sum(x["compressed_background_tokens"] for x in successes) if successes else None}
        (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Finished sample {idx}: {result['status']} in {result['elapsed_seconds']}s; completed {len(done)}/{len(samples)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "evaluate"])
    p.add_argument("--outdir", type=Path, default=ROOT / "experiments/long-code-completion/results/awq-pilot")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--test-parquet", type=Path, default=Path("/tmp/lcz_lcc_test.parquet"))
    p.add_argument("--target-tokens", type=int, default=2048)
    p.add_argument("--fine-ratio", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    p.add_argument("--cpu-threads", type=int, default=4)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args.outdir, args.count, args.test_parquet)
    else:
        evaluate(args.outdir, args)
