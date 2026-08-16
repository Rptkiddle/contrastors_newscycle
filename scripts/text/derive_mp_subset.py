"""Derive a smaller-N multi-positive shard set from a mined larger-N one.

The gdelter build emits top-N positives in cascade order with a `rank`
field, so the mp3 dataset is exactly the rank < 3 subset of the mined mp5
records — same queries, same negatives. Filtering the mined shards avoids
a second mining run.

    python derive_mp_subset.py --input_dir mp5/ --output_dir mp3/ --max_rank 3
"""

import argparse
import glob
import gzip
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rank", type=int, required=True,
                        help="keep records with rank < max_rank")
    parser.add_argument("--shard_size", type=int, default=100_000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(Path(args.input_dir) / "shard-*.jsonl.gz")))
    if not shards:
        raise FileNotFoundError(f"no shards in {args.input_dir}")

    n_in = n_out = shard_num = 0
    buf = []

    def flush():
        nonlocal shard_num
        final = out / f"shard-{shard_num:05d}.jsonl.gz"
        tmp = out / f"shard-{shard_num:05d}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt") as f:
            for line in buf:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        shard_num += 1
        buf.clear()

    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                n_in += 1
                if json.loads(line)["rank"] < args.max_rank:
                    buf.append(line)
                    n_out += 1
                    if len(buf) >= args.shard_size:
                        flush()
    if buf:
        flush()

    summary = {"input_dir": args.input_dir, "max_rank": args.max_rank,
               "records_in": n_in, "records_out": n_out, "shards": shard_num}
    with open(out / "derive_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
