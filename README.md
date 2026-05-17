# BPE Tokenizer From Scratch

A minimal Byte Pair Encoding (BPE) tokenizer built from scratch in Python to understand how modern LLM tokenization works internally.

## Why This Exists

Large language models like GPT-2 and Llama don't process raw text directly — they operate on sequences of integer token IDs. Before text reaches a model, it must be tokenized into smaller reusable units.

The goal of this project was to understand tokenization conceptually rather than treat it as a black box. This implementation recreates the core mechanics of BPE from scratch: learning merges from training text, building a vocabulary dynamically, encoding unseen text using learned merges, and decoding token IDs back into the original text.

The focus is educational clarity rather than production optimization.

## Files

| File | Description |
| --- | --- |
| `bpe.py` | BPE tokenizer class — training, encoding, decoding |
| `benchmark.py` | Benchmarks custom BPE against tiktoken cl100k_base |
| `results/benchmark_results.md` | Latest benchmark output |

## How BPE Works

BPE starts with individual bytes as tokens, then repeatedly:

1. Counts neighboring token pairs
2. Finds the most frequent pair
3. Merges it into a new token
4. Rewrites the sequence

Over time, common patterns become reusable chunks:

After training, encoding simply replays the learned merges in the same order.

## Benchmark Results

Trained on Project Gutenberg books (~750k chars), tested on a held-out split.

| tokenizer | vocab | train (s) | tokens | chars/tok | oov % | k tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| Custom BPE | 5,256 | 1,725 | 191,308 | 3.961 | 8.71% | 0.6 |
| tiktoken cl100k_base | 100,277 | — | 186,327 | 4.067 | — | 880 |

**Compression** is competitive — 3.96 vs 4.07 chars/token — despite using a 19× smaller vocabulary. The speed gap (0.6 vs 880 k tok/s) is pure Python vs compiled C.

## Setup

```bash
git clone <repo-url>
cd BPE_Tokenizer
pip install tiktoken datasets   # datasets optional — benchmark falls back to Gutenberg
```

Python 3.10+ required.

## Usage

```python
from bpe import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train("hello hello hello world", vocab_size=300)

ids  = tokenizer.encode("hello")
text = tokenizer.decode(ids)

assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
```

## Run the Benchmark

```bash
python benchmark.py               # 5000 merges (~30 min)
python benchmark.py --merges 1000 # faster smoke test
```

Results are saved to `results/benchmark_results.md`.

## What I'd Do Next

- Regex-based pre-tokenization (GPT-2 style)
- Priority queue for faster merge selection
- Special tokens (`<PAD>`, `<EOS>`, etc.)
- Serialize vocab and merges to disk
- Match GPT-2 byte encoding exactly

## References

- Andrej Karpathy — "Let's build the GPT Tokenizer"
- Sennrich et al. — *Neural Machine Translation of Rare Words with Subword Units*
- [OpenAI TikToken](https://github.com/openai/tiktoken)
- [Hugging Face Tokenizers Docs](https://huggingface.co/docs/tokenizers)