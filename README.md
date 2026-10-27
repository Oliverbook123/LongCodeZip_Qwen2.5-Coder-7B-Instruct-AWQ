<div align="center">
  <img src="assets/logo.png" alt="LongCodeZip Logo" width="200"/>

[![arXiv](https://img.shields.io/badge/arXiv-2510.00446-b31b1b.svg)](https://arxiv.org/abs/2510.00446) [![Accepted: ASE 2025](https://img.shields.io/badge/Accepted-ASE%202025-brightgreen.svg)](https://conf.researchr.org/details/ase-2025/ase-2025-papers/121/LongCodeZip-Compress-Long-Context-for-Code-Language-Models) [![Python Version](https://img.shields.io/badge/Python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/) [![GitHub stars](https://img.shields.io/github/stars/YerbaPage/LongCodeZip?style=social)](https://github.com/YerbaPage/LongCodeZip) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

# LongCodeZip

This repository is the official implementation of LongCodeZip, a novel two-stage long code compression method. Our paper "LongCodeZip: Compress Long Context for Code Language Models" has been accepted to **ASE 2025**.

> This fork adds [`demo_awq.py`](demo_awq.py), a single-model AWQ demo for GPUs with limited VRAM. The original upstream repository is [YerbaPage/LongCodeZip](https://github.com/YerbaPage/LongCodeZip).

## Method Overview

![Overview](assets/overview.png)

LongCodeZip introduces a two-stage code compression framework specifically designed for code LLMs:

1. **Coarse-grained Compression**: Function-based chunking and ranking using conditional perplexity with respect to the query to select the most relevant functions.

2. **Fine-grained Compression**: Entropy-based block detection combined with 0/1 knapsack optimization to maximize relevance within adaptive token budgets.

The method is plug-and-play and can be integrated with existing code LLMs to achieve significant compression ratios while maintaining or improving task performance.

## Installation

You can install directly from the GitHub repository:

```bash
pip install git+https://github.com/Oliverbook123/LongCodeZip_Qwen2.5-Coder-7B-Instruct-AWQ.git
```

Or clone and install in development mode:

```bash
git clone https://github.com/Oliverbook123/LongCodeZip_Qwen2.5-Coder-7B-Instruct-AWQ.git
cd LongCodeZip_Qwen2.5-Coder-7B-Instruct-AWQ
pip install -e .
```

## Quick Demo

Two demos are provided:

| Script | Compression model | Generation model | Notes |
|---|---|---|---|
| `demo.py` | `Qwen/Qwen2.5-Coder-7B-Instruct` | separate 7B model or OpenAI API | Loads two large models; needs ~30GB if both 7B models are local |
| `demo_awq.py` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | the same AWQ model | Reuses one model for both stages; suitable for 8GB GPUs |

### Original Demo

```bash
python demo.py
```

The demo showcases both compression modes: coarse-grained compression (function-level selection only) and the full two-stage compression (with fine-grained token optimization). It demonstrates how LongCodeZip compresses a code file based on a given query and achieves different compression ratios.

After compression, the demo uses `Qwen/Qwen2.5-Coder-7B-Instruct` as the default generation model.  
You can change this by specifying `--generation_model`.

To use an OpenAI API instead of an open source model, add the `--use_openai` flag. 

**Example:**
```bash
python demo.py --use_openai --generation_model gpt-4o
```
It is recommended to set your API key as an environment variable. For example, create a `.env` file and add: `OPENAI_API_KEY="enter_ your_key_here"`. The OpenAI path also requires `pip install python-dotenv`.

### Single-Model AWQ Demo (8GB-friendly)

`demo.py` keeps the compression model and the generation model in VRAM at the same time. On a small GPU, loading two 7B models is usually impossible. `demo_awq.py` avoids this by using `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` for **both compression and generation**:

- one AWQ model instance is shared by both stages;
- both compression modes run before generation;
- the internal 0.5B entropy model is freed after compression;
- generation reuses the already-loaded AWQ model instead of loading a second 7B model.

Install the AWQ runtime:

```bash
pip install autoawq
```

Run:

```bash
python demo_awq.py
```

Useful options:

```bash
python demo_awq.py --device_map auto
python demo_awq.py --max_new_tokens 64
python demo_awq.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct  # test without autoawq
```

You can also use the AWQ model directly through the library:

```python
import torch
from longcodezip import LongCodeZip

compressor = LongCodeZip(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    model_config={"torch_dtype": torch.float16},
)
```

**Memory notes:**

- The AWQ checkpoint is about 5.6GB. Together with the 0.5B entropy model, the demo uses roughly 6.6GB before compression activations.
- 8GB is tight: short code files can work, but very long functions may still run out of memory.
- `--device_map auto` may offload some layers to CPU and is slower.
- The entropy chunking model remains `Qwen/Qwen2.5-Coder-0.5B-Instruct`; only the main compression/generation model is AWQ.
- `autoawq` is distributed mainly as a source package. If it cannot be installed for your torch/CUDA version, test the demo with `--model Qwen/Qwen2.5-Coder-0.5B-Instruct`.

## Basic Example

```python
from longcodezip import LongCodeZip

# Initialize the compressor
compressor = LongCodeZip(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

# Compress code with a query
result = compressor.compress_code_file(
    code=<your_code_string>,
    query=<your_query>,
    instruction=<your_instruction>,
    rate=0.5,  # Keep 50% of tokens
    rank_only=False, # Set to True to only rank and select contexts without fine-grained compression
)

# Access compressed results
compressed_code = result['compressed_code']
compressed_prompt = result['compressed_prompt']  # Full prompt with instruction
compression_ratio = result['compression_ratio']
```

## References

```bibtex
@article{shi2025longcodezip,
  title={LongCodeZip: Compress Long Context for Code Language Models},
  author={Shi, Yuling and Qian, Yichun and Zhang, Hongyu and Shen, Beijun and Gu, Xiaodong},
  journal={arXiv preprint arXiv:2510.00446},
  year={2025}
}
```
