#!/usr/bin/env python3
"""
Generate counts.json and offsets.json.gz for a set of .jsonl.gz shards.

This script reads each gzip-compressed JSONL shard, records the uncompressed byte
start/end offsets for each line (record), and writes two files:

- counts.json: mapping normalized_filename -> number_of_examples
- offsets.json.gz: mapping normalized_filename -> { index: [start, end], ... }

Normalization follows the loader's `normalize_url` behavior (keep last 3-4 path
components) so the produced keys match what the dataset loader expects.

Usage:
    python scripts/text/generate_counts_offsets.py --shard-pattern '/path/to/shard-{00000..00001}.jsonl.gz' \
        --out-dir /path/to/output_parent_dir

If your pattern uses s3:// or local paths, the script will expand the pattern
using webdataset's shardlist expander if available; otherwise it will try to
treat the pattern as a glob.
"""
import argparse
import gzip
import json
import os
from pathlib import Path

try:
    import webdataset as wds
except Exception:
    wds = None


def normalize_url_single(url: str) -> str:
    """Normalize like StreamingShardDataset.normalize_url: keep last 3-4 parts."""
    parts = url.split("/")
    if len(parts) >= 6:
        return "/".join(parts[-4:])
    else:
        return "/".join(parts[-3:])


def expand_pattern(pattern: str):
    # prefer webdataset expansion if available (handles {00000..00004})
    if wds is not None:
        try:
            return list(wds.shardlists.expand_urls(pattern))
        except Exception:
            pass

    # fallback to simple brace expansion for local patterns like shard-{00000..00004}.jsonl.gz
    if "{" in pattern and ".." in pattern and "}" in pattern:
        prefix, rest = pattern.split("{", 1)
        range_part, suffix = rest.split("}", 1)
        start_str, end_str = range_part.split("..")
        start = int(start_str)
        end = int(end_str)
        width = max(len(start_str), len(end_str))
        return [f"{prefix}{i:0{width}d}{suffix}" for i in range(start, end + 1)]

    # last resort: glob
    import glob

    return glob.glob(pattern)


def process_shard(path: str):
    offsets = {}
    count = 0
    # open gzip in binary mode; gzip.GzipFile provides tell() in uncompressed bytes
    with gzip.open(path, "rb") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            end = f.tell()
            offsets[str(count)] = [start, end]
            count += 1

    return count, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-pattern", type=str, required=True, help="Shard pattern or comma-separated list of files")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory where counts.json and offsets.json.gz will be written")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pattern = args.shard_pattern
    # Support comma separated list
    if "," in pattern:
        paths = [p.strip() for p in pattern.split(",") if p.strip()]
    else:
        paths = expand_pattern(pattern)

    if not paths:
        raise SystemExit(f"No shards found for pattern: {pattern}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    offsets_out = {}

    for p in paths:
        if args.verbose:
            print(f"Processing shard: {p}")
        if not os.path.exists(p):
            raise SystemExit(f"Shard not found: {p}")

        count, offsets = process_shard(p)
        norm = normalize_url_single(p)
        counts[norm] = count
        offsets_out[norm] = offsets
        if args.verbose:
            print(f"  lines: {count}")

    # write counts.json (uncompressed)
    counts_path = out_dir / "counts.json"
    with open(counts_path, "w") as f:
        json.dump(counts, f)

    # write offsets.json.gz
    offsets_path = out_dir / "offsets.json.gz"
    with gzip.open(offsets_path, "wb") as f:
        f.write(json.dumps(offsets_out).encode("utf-8"))

    print(f"Wrote counts to {counts_path}")
    print(f"Wrote offsets to {offsets_path}")


if __name__ == "__main__":
    main()
