"""Demo: use one Qwen2.5-Coder AWQ model for both compression and generation.

This is a VRAM-conscious variant of demo.py:

1. LongCodeZip loads the compression model and a 0.5B entropy model.
2. The same compression model is reused as the generation model.
3. Both compression modes are run first, then the entropy model is freed.
4. Generation uses the already-loaded main model, so only one 7B model is resident.

Requirements:
    pip install autoawq

For the default AWQ model, autoawq is required because transformers delegates
AWQ loading to that package. On an 8GB GPU, the AWQ weights alone are ~5.6GB;
this demo can work for short contexts but long functions may still OOM.

If autoawq cannot be installed, the script can be tested with a smaller model:
    python demo_awq.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct
"""

import argparse
import gc
import re

import torch
from loguru import logger

from longcodezip import LongCodeZip

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"


def gpu_mem(tag: str) -> None:
    """Log allocated/reserved GPU memory for easier 8GB debugging."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(
        f"[GPU] {tag}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"
    )


def load_compressor(model_name: str, device_map: str) -> LongCodeZip:
    try:
        return LongCodeZip(
            model_name=model_name,
            device_map=device_map,
            # AWQ checkpoints are stored for fp16 compute.
            model_config={"torch_dtype": torch.float16},
        )
    except ImportError as exc:
        message = str(exc).lower()
        if "awq" in message:
            raise SystemExit(
                "Loading the AWQ model requires the 'autoawq' package.\n"
                "Install it first:\n"
                "    pip install autoawq\n\n"
                "If autoawq cannot be installed for this torch/CUDA version, test "
                "the demo with a smaller non-AWQ model:\n"
                "    python demo_awq.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct"
            ) from exc
        raise


def generate_completion(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> str:
    """Greedy generation using the same model instance as compression."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Remove markdown code fences if the model adds them.
    if generated.startswith("```"):
        generated = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", generated)
        generated = re.sub(r"\s*```$", "", generated)

    return generated.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongCodeZip demo with one shared Qwen2.5-Coder AWQ model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model used for both compression and generation.",
    )
    parser.add_argument(
        "--device_map",
        default="cuda",
        help="Transformers device_map, e.g. cuda or auto.",
    )
    parser.add_argument(
        "--context_file",
        default="assets/example_context.py",
        help="Code file to compress.",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=64,
        help="Target token budget for coarse-grained compression.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    with open(args.context_file, "r", encoding="utf-8") as handle:
        context = handle.read()

    question = '''
    async def _finalize_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> None:
        step.state = AgentStepState.COMPLETED
    '''
    instruction = "Complete the following code function given the context."

    logger.info(f"Loading model: {args.model}")
    compressor = load_compressor(args.model, args.device_map)

    # Reuse the compressor's model and tokenizer for generation. Loading a
    # second 7B model here would almost certainly OOM on an 8GB GPU.
    model = compressor.model
    tokenizer = compressor.tokenizer
    gpu_mem("after loading compression model + entropy model")

    original_tokens = len(tokenizer.encode(context))
    target_ratio = min(1.0, max(0.0, args.target_tokens / original_tokens))
    logger.info(
        f"Original tokens={original_tokens}, target tokens={args.target_tokens}, "
        f"calculated ratio={target_ratio:.4f}"
    )

    # Run both compressions before generating. This ensures the entropy model
    # is not needed while generation is running.
    logger.info("Running coarse-grained compression...")
    coarse_result = compressor.compress_code_file(
        code=context,
        query=question,
        instruction=instruction,
        rate=target_ratio,
        rank_only=True,
    )
    logger.info(f"Coarse compressed prompt:\n{coarse_result['compressed_prompt']}")
    logger.info(f"Coarse compression ratio: {coarse_result['compression_ratio']:.4f}")

    logger.info("Running coarse-grained + fine-grained compression...")
    fine_result = compressor.compress_code_file(
        code=context,
        query=question,
        instruction=instruction,
        rate=target_ratio,
        rank_only=False,
    )
    logger.info(f"Fine-grained compressed prompt:\n{fine_result['compressed_prompt']}")
    logger.info(f"Fine-grained compression ratio: {fine_result['compression_ratio']:.4f}")

    # The 0.5B entropy model is only used by fine-grained compression. Free it
    # before generation to leave as much VRAM as possible for the AWQ model.
    del compressor.entropy_chunking
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem("after freeing entropy model")

    for name, result in (
        ("coarse-grained", coarse_result),
        ("coarse+fine-grained", fine_result),
    ):
        logger.info(f"Generating with {args.model} ({name})...")
        completion = generate_completion(
            prompt=result["compressed_prompt"],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        logger.info(f"Completion ({name}):\n{completion}")

    del model
    del tokenizer
    del compressor
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem("after cleanup")


if __name__ == "__main__":
    main()
