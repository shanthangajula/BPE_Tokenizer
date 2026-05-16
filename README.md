BPE Tokenizer From Scratch

A minimal Byte Pair Encoding (BPE) tokenizer implementation built from scratch in Python to understand how modern LLM tokenization works internally.

Why This Exists

Large language models such as GPT-2 and Llama do not process raw text directly. They operate on sequences of integer token IDs. Before text can be fed into a model, it must first be tokenized into smaller reusable units.

The goal of this project was to understand tokenization conceptually instead of treating it as a black box. Rather than using an existing tokenizer library, this implementation recreates the core mechanics of Byte Pair Encoding (BPE) from scratch:

learning merges from training text
building a vocabulary dynamically
encoding unseen text using learned merges
decoding token IDs back into the original text

The focus of this project is educational clarity rather than production optimization.

What's In Here

This repository contains:

tokenizer.py
Minimal BPE tokenizer implementation
Byte-level vocabulary initialization
BPE training loop
Encoding and decoding logic
test_tokenizer.py
Simple examples for training and testing the tokenizer
README.md
Explanation of the project and implementation details

Core features implemented:

Byte-level tokenization
Adjacent pair frequency counting
Iterative merge learning
Deterministic encoding using learned merge order
Lossless decoding back to UTF-8 text
How BPE Works (30-Second Version)

BPE starts with individual bytes as tokens.

Example:

"h e l l o"

The tokenizer repeatedly:

Counts neighboring token pairs
Finds the most frequent pair
Merges that pair into a new token
Rewrites the sequence using the new token

Over time, common patterns become reusable chunks:

"h" + "e" -> "he"
"he" + "llo" -> "hello"

Frequent sequences become compressed into larger tokens, while rare words can still be represented using smaller byte-level pieces.

After training, encoding simply replays the learned merges in the same order.

Results

Example training corpus:

"hello hello hello world"

Example run:

Step	Output
Encoded "hello"	[259]
Decoded Output	"hello"
Vocabulary Size	300
Base Vocabulary	256 byte tokens
Learned Merges	44

The tokenizer successfully satisfies:

decode(encode(text)) == text

which confirms lossless reconstruction.

Setup

Clone the repository:

git clone <repo-url>
cd <repo-name>

Python 3.10+ recommended.

No external dependencies are required beyond the Python standard library.

Usage

Train tokenizer:

from tokenizer import BPETokenizer

tokenizer = BPETokenizer()

corpus = "hello hello hello world"

tokenizer.train(corpus, vocab_size=300)

Encode text:

ids = tokenizer.encode("hello")

print(ids)

Decode token IDs:

text = tokenizer.decode(ids)

print(text)

Expected property:

tokenizer.decode(
    tokenizer.encode("hello")
) == "hello"
What I'd Do Next

This implementation focuses on conceptual understanding and correctness, but there are several improvements that would make it closer to production-grade tokenizers:

Add regex-based pre-tokenization
Store merge rankings explicitly
Improve merge efficiency using priority queues
Add special tokens (<PAD>, <EOS>, etc.)
Serialize vocab and merges to disk
Benchmark compression ratio and speed
Support larger training corpora
Implement GPT-2 style byte encoding exactly

A future extension would also be comparing this tokenizer against real-world implementations from OpenAI and Hugging Face.

References
Andrej Karpathy — “Let’s build the GPT Tokenizer”
GPT-2 tokenizer implementation
Sennrich et al. — Neural Machine Translation of Rare Words with Subword Units
Hugging Face Tokenizers Documentation
OpenAI TikToken Repository