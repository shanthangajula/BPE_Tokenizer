# BPE Benchmark Results

_Generated: 2026-05-17 08:31:20_

## Metrics
- **chars/tok** — compression ratio; higher = fewer tokens needed
- **oov %** — raw single-byte tokens; lower = better vocab coverage
- **k tok/s** — encoding speed on the holdout set

## Results
| tokenizer | vocab | train (s) | tokens | chars/tok | oov % | k tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| Custom BPE | 5256 | 1725.71 | 191308 | 3.961 | 8.71 | 0.6 |
| tiktoken cl100k_base | 100277 | — | 186327 | 4.067 | — | 880.2 |
