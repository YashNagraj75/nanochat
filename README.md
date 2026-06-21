# nanochat

> **Work in Progress** — This project is not yet complete. Core components are still under active development and the chat interface is not functional yet.

A from-scratch GPT-based language model and chat interface built in Python. The project implements core components of a transformer language model — starting with a tokenizer — targeting CUDA-accelerated training and inference via PyTorch.

## Features

- Custom tokenizer implementation under `gpt/`
- PyTorch-based model with CUDA / Triton acceleration
- Tokenization via `tiktoken` and `tokenizers` (HuggingFace)
- CLI interface powered by Typer and Rich
- HuggingFace Hub integration for model weights

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Language | Python 3.14 |
| Deep learning | PyTorch 2.11, Triton 3.6 |
| CUDA runtime | CUDA 13 (cublas, cudnn, curand, cusolver, nccl, nvjitlink) |
| Tokenization | tiktoken 0.12, tokenizers 0.23, rustbpe |
| Model hub | huggingface-hub |
| CLI | Typer, Rich |
| Package manager | uv |

## Project Structure

```
nanochat/
├── gpt/
│   └── tokenizer.py     # Tokenizer implementation
├── main.py              # CLI entry point
├── pyproject.toml       # Project metadata and build config
├── requirements.txt     # Pinned dependency lockfile
└── uv.lock              # uv resolver lockfile
```

## Installation

### Prerequisites

- Python 3.14
- A CUDA-capable GPU (CUDA 13 toolkit)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YashNagraj75/nanochat.git
cd nanochat

# Create a virtual environment and install dependencies with uv
uv sync

# Or install with pip from the pinned requirements
pip install -r requirements.txt
```

## Usage

Run the main entry point:

```bash
python main.py
```

Or, if installed as a package:

```bash
uv run nanochat
```

## Development Status

This project is a **work in progress and is not yet complete**. The following components are still being built:

- Transformer model (encoder/decoder blocks, attention, positional embeddings)
- Prefill and decode stages for the inference engine
- Chat interface and interactive CLI

The `gpt/` package currently contains only the tokenizer stub. The broader model architecture and chat loop are under active development.

## Examples

_Screenshots and usage examples will be added as the chat interface matures._
