"""
BPE Benchmark
=============
Trains BPETokenizer on wikitext-2 (train split), encodes the test split,
and compares against tiktoken cl100k_base on the same holdout.

Metrics
-------
  compression_ratio   chars / token  (higher = more compression)
  oov_rate            fraction of tokens that are raw single bytes (<256)
  train_time_s        wall-clock seconds to train
  encode_speed        k tokens / second on the holdout

Usage
-----
  python benchmark.py               # 5000 merges (default)
  python benchmark.py --merges 1000 # faster smoke test
"""

import argparse
import glob
import time

import tiktoken
from bpe import BPETokenizer


# ── corpus ────────────────────────────────────────────────────────────────────

def load_corpus() -> tuple[str, str]:
    """Return (train_text, test_text). Tries HuggingFace, then local fallback."""
    try:
        from datasets import load_dataset
        print("Loading wikitext-2 from HuggingFace ...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train = "\n".join(ds["train"]["text"])
        test  = "\n".join(ds["test"]["text"])
        print(f"  train {len(train):,} chars / test {len(test):,} chars")
        return train, test
    except Exception as e:
        print(f"HuggingFace unavailable ({e}), using local stdlib fallback")
        return _local_fallback()


def _local_fallback() -> tuple[str, str]:
    """Read Python stdlib .py files as a plain-English prose substitute."""
    files = sorted(glob.glob("/usr/lib/python3.*/**/*.py", recursive=True))
    texts = []
    for path in files:
        try:
            texts.append(open(path, errors="replace").read())
        except OSError:
            pass
    if not texts:
        raise RuntimeError("No corpus available — install the `datasets` package.")
    split = int(len(texts) * 0.9)
    train = "\n\n".join(texts[:split])
    test  = "\n\n".join(texts[split:])
    print(f"  {len(texts)} stdlib files  |  train {len(train):,} chars / test {len(test):,} chars")
    return train, test


# ── metrics ───────────────────────────────────────────────────────────────────

def compression_ratio(text: str, ids: list[int]) -> float:
    return len(text) / len(ids) if ids else 0.0


def oov_rate(ids: list[int]) -> float:
    """Fraction of tokens that are raw single bytes — indicates poor coverage."""
    return sum(1 for t in ids if t < 256) / len(ids) if ids else 0.0


# ── benchmark runs ────────────────────────────────────────────────────────────

def run_custom_bpe(train_text: str, test_text: str, vocab_size: int) -> dict:
    print(f"\n── Custom BPE  (vocab_size={vocab_size}) {'─'*40}")
    tok = BPETokenizer()

    t0 = time.perf_counter()
    tok.train(train_text, vocab_size=vocab_size)
    train_time = time.perf_counter() - t0
    print(f"  trained in {train_time:.1f}s  |  merges learned: {len(tok.merges)}")

    t1 = time.perf_counter()
    ids = tok.encode(test_text)
    encode_time = time.perf_counter() - t1

    cr   = compression_ratio(test_text, ids)
    oov  = oov_rate(ids)
    speed = len(ids) / encode_time / 1_000 if encode_time > 0 else float("nan")
    print(f"  tokens: {len(ids):,}  |  ratio: {cr:.3f}  |  OOV: {oov*100:.1f}%  |  speed: {speed:.0f} k tok/s")

    return {
        "name":             "Custom BPE",
        "vocab_size":       len(tok.vocab),
        "train_time_s":     round(train_time, 2),
        "test_tokens":      len(ids),
        "compression_ratio": round(cr, 3),
        "oov_pct":          round(oov * 100, 2),
        "encode_ktps":      round(speed, 1),
    }


def run_tiktoken(test_text: str) -> dict:
    print(f"\n── tiktoken  cl100k_base {'─'*50}")
    enc = tiktoken.get_encoding("cl100k_base")

    t0 = time.perf_counter()
    ids = enc.encode(test_text)
    encode_time = time.perf_counter() - t0

    cr    = compression_ratio(test_text, ids)
    speed = len(ids) / encode_time / 1_000 if encode_time > 0 else float("nan")
    print(f"  tokens: {len(ids):,}  |  ratio: {cr:.3f}  |  speed: {speed:.0f} k tok/s")

    return {
        "name":             "tiktoken cl100k_base",
        "vocab_size":       enc.n_vocab,
        "train_time_s":     "—",
        "test_tokens":      len(ids),
        "compression_ratio": round(cr, 3),
        "oov_pct":          "—",
        "encode_ktps":      round(speed, 1),
    }


# ── summary table ─────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    cols = ["name", "vocab_size", "train_time_s", "test_tokens",
            "compression_ratio", "oov_pct", "encode_ktps"]
    headers = ["tokenizer", "vocab", "train (s)", "tokens",
               "chars/tok", "oov %", "k tok/s"]
    widths = [max(len(h), max(len(str(r[c])) for r in results))
              for h, c in zip(headers, cols)]

    sep = "  ".join("─" * w for w in widths)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(f"\n{'─'*len(sep)}")
    print(row_fmt.format(*headers))
    print(sep)
    for r in results:
        print(row_fmt.format(*[str(r[c]) for c in cols]))
    print(f"{'─'*len(sep)}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merges", type=int, default=5_000,
                        help="BPE vocab size = 256 + merges (default: 5000)")
    args = parser.parse_args()

    train_text, test_text = load_corpus()

    results = [
        run_custom_bpe(train_text, test_text, vocab_size=256 + args.merges),
        run_tiktoken(test_text),
    ]

    print_table(results)


if __name__ == "__main__":
    main()