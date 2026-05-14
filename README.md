# BPE_Tokenizer
A from-scratch BPE tokenizer in Python. Includes training, encoding, decoding, and benchmarks against tiktoken (compression ratio, OOV rate, training time).

Install — pip install tiktoken datasets (the only two deps)
Quick start — 4-line code snippet showing train → encode → decode
Run the benchmark — the python benchmark.py --merges 1000 command and what the output table looks like
File layout — bpe.py (the class) vs benchmark.py (the harness), one line each