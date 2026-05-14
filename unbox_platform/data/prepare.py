"""Download and cache FineWeb-Edu sample-10BT to disk as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def download_fineweb_edu(
    output_path: Path,
    subset: str = "sample-10BT",
    cache_dir: str = "data/cache",
) -> None:
    from datasets import load_dataset

    print(f"Downloading FineWeb-Edu {subset} ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc="Writing JSONL"):
            text = example.get("text", "")
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count:,} documents to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu to local JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/fineweb_edu_10bt.jsonl"))
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    download_fineweb_edu(args.output, subset=args.subset, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
