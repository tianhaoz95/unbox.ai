"""Stream 5B tokens from AI-ModelScope/fineweb-edu and write to JSONL.

ModelScope does not host the filtered subsets (e.g. sample-10BT), only the
full dataset. This script streams documents one at a time and stops as soon as
the cumulative token count reaches TARGET_TOKENS, so only the needed data ever
touches disk.

Each document in FineWeb-Edu carries a `token_count` field (GPT-2 tokens,
computed by the FineWeb team). We use that directly. If the field is absent for
a document, we fall back to len(text) // 4 as a rough estimate.

Usage:
    .venv/bin/python scripts/download_fineweb_5b.py
    .venv/bin/python scripts/download_fineweb_5b.py --output data/fineweb_5b.jsonl
    .venv/bin/python scripts/download_fineweb_5b.py --target-tokens 2_000_000_000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TARGET_TOKENS_DEFAULT = 5_000_000_000
DATASET_NAME = "AI-ModelScope/fineweb-edu"
LOG_INTERVAL = 100_000  # documents between progress prints


def _fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def download(output_path: Path, target_tokens: int, cache_dir: str) -> None:
    from modelscope.msdatasets import MsDataset  # type: ignore[import]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_NAME} → {output_path}")
    print(f"Target: {_fmt(target_tokens)} tokens")

    ds = MsDataset.load(
        DATASET_NAME,
        split="train",
        use_streaming=True,
        cache_dir=cache_dir,
    )

    total_tokens = 0
    total_docs = 0
    start = time.monotonic()

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "")
            if not text:
                continue

            doc_tokens: int = doc.get("token_count") or (len(text) // 4)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_tokens += doc_tokens
            total_docs += 1

            if total_docs % LOG_INTERVAL == 0:
                elapsed = time.monotonic() - start
                pct = 100.0 * total_tokens / target_tokens
                rate = total_tokens / elapsed if elapsed > 0 else 0
                eta = (target_tokens - total_tokens) / rate if rate > 0 else float("inf")
                print(
                    f"  {_fmt(total_tokens)} tokens ({pct:.1f}%)  "
                    f"{total_docs:,} docs  "
                    f"{_fmt(int(rate))} tok/s  "
                    f"ETA {eta / 60:.0f}m",
                    flush=True,
                )

            if total_tokens >= target_tokens:
                break

    elapsed = time.monotonic() - start
    print(f"\nDone: {_fmt(total_tokens)} tokens, {total_docs:,} docs in {elapsed / 60:.1f}m")
    size_mb = output_path.stat().st_size / 1024**2
    print(f"Output: {output_path}  ({size_mb:.0f} MB)")

    if total_tokens < target_tokens:
        print(
            f"WARNING: dataset exhausted at {_fmt(total_tokens)} tokens "
            f"(target was {_fmt(target_tokens)})",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ~5B tokens from FineWeb-Edu via ModelScope")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fineweb_edu_5b.jsonl"),
        help="Output JSONL path (default: data/fineweb_edu_5b.jsonl)",
    )
    parser.add_argument(
        "--target-tokens",
        type=lambda s: int(s.replace("_", "")),
        default=TARGET_TOKENS_DEFAULT,
        metavar="N",
        help="Stop after this many tokens (default: 5_000_000_000)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="ModelScope download cache directory",
    )
    args = parser.parse_args()

    download(args.output, args.target_tokens, args.cache_dir)


if __name__ == "__main__":
    main()
