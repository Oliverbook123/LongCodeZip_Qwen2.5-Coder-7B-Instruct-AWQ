"""Run the LongCodeZip demo using vLLM for AWQ scoring and generation.

The entropy model runs on CPU to leave GPU memory for the 7B model. This
adapter uses vLLM prompt log probabilities for the original PPL calculations.
"""
import argparse
import gc
import math
import os

# Prompt logprob calculation otherwise requires a system C compiler.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from loguru import logger
from vllm import LLM, SamplingParams

from longcodezip import EntropyChunking, LongCodeZip


class VllmCompressor(LongCodeZip):
    def __init__(self, model_name, max_model_len, gpu_memory_utilization, load_entropy=True,
                 cpu_offload_gb=0):
        self.model_name = model_name
        self.device = "cpu"
        self.model_config = {}
        self.max_position_embeddings = max_model_len
        if load_entropy:
            self.entropy_chunking = EntropyChunking()
            self.entropy_chunking.model.to("cpu")
            self.entropy_chunking.device = torch.device("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = LLM(
            model=model_name,
            quantization="awq_marlin",
            max_model_len=max_model_len,
            max_num_batched_tokens=128,
            max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=True,
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _losses(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) + 1 > self.max_position_embeddings:
            raise ValueError(f"Input has {len(ids)} tokens; vLLM limit is {self.max_position_embeddings - 1}.")
        if len(ids) < 2:
            return ids, []
        result = self.engine.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1),
            use_tqdm=False,
        )[0]
        return ids, [-result.prompt_logprobs[i][token].logprob
                     for i, token in enumerate(ids) if i > 0]

    def get_ppl(self, text, granularity="line", input_ids=None, attention_mask=None,
                past_key_values=None, return_kv=False, end=None,
                condition_mode="none", condition_pos_id=0):
        if input_ids is not None or past_key_values is not None or return_kv or end is not None:
            raise NotImplementedError("The vLLM demo does not support cached or partial inputs.")
        ids, losses = self._losses(text)
        if condition_mode == "prefix":
            losses = losses[condition_pos_id:]
        loss = torch.tensor(losses)
        ppl = math.exp(loss.mean().item()) if losses else 1.0
        return {"loss": loss, "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                "lines_info": [], "segments": [text] if text else [], "ppl": ppl}

    def _calculate_perplexity_for_contrastive(self, text, condition_text=None):
        _, losses = self._losses((condition_text or "") + text)
        if condition_text:
            prefix_length = len(self.tokenizer.encode(condition_text, add_special_tokens=True))
            losses = losses[max(0, prefix_length - 1):]
        return math.exp(sum(losses) / len(losses)) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser(description="LongCodeZip demo using vLLM AWQ")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    parser.add_argument("--context_file", default="assets/example_context.py")
    parser.add_argument("--target_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.78)
    args = parser.parse_args()
    with open(args.context_file, encoding="utf-8") as handle:
        context = handle.read()
    question = '''
    async def _finalize_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> None:
        step.state = AgentStepState.COMPLETED
    '''
    instruction = "Complete the following code function given the context."
    logger.info(f"Loading vLLM model: {args.model}")
    compressor = VllmCompressor(args.model, args.max_model_len, args.gpu_memory_utilization)
    original_tokens = len(compressor.tokenizer.encode(context))
    rate = min(1.0, max(0.0, args.target_tokens / original_tokens))
    logger.info(f"Original tokens={original_tokens}, target={args.target_tokens}, ratio={rate:.4f}")
    for name, rank_only in (("coarse", True), ("coarse+fine", False)):
        logger.info(f"Running {name} compression...")
        result = compressor.compress_code_file(code=context, query=question,
            instruction=instruction, rate=rate, rank_only=rank_only)
        logger.info(f"Compressed prompt ({name}):\n{result['compressed_prompt']}")
        logger.info(f"Compression ratio ({name}): {result['compression_ratio']:.4f}")
        output = compressor.engine.generate(
            [result["compressed_prompt"]],
            SamplingParams(temperature=0, max_tokens=args.max_new_tokens), use_tqdm=False)[0]
        logger.info(f"Completion ({name}):\n{output.outputs[0].text.strip()}")
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
