"""Mine hard negatives for NewsCycle training splits.

For each training record (one query and one positive document per
(entity, month-year) key), the candidate pool is every other record in
the same split with the same normalised entity and a different
month-year. Candidates are ranked by gte-base query-document cosine
similarity (the miner Nomic used for their fine-tuning datasets); the
top --k survivors are stored as hard negatives.

Two near-duplicate screens run before selection, both using the same
5-gram crc32 shingle Jaccard machinery and threshold as the data
pipeline (newscycle-gdelter/process.py):
  1. a candidate that near-duplicates the positive is excluded
     (wire copy re-run in a different month is not a valid negative);
  2. a candidate that near-duplicates an already selected negative is
     excluded (the stored list holds distinct articles).

Records with fewer than --min-negatives surviving candidates are
dropped: the contrastive loss assumes a uniform negative count within a
batch, so short records would silently misalign the labels. Drop counts
and screen statistics are written to mining_stats.json.

Run on a single GPU:
    python get_negatives_news.py --dataset train_merged.jsonl.gz --output_dir out/
"""

import argparse
import gzip
import json
import os
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "thenlper/gte-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="train_*.jsonl or .jsonl.gz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20, help="negatives stored per record")
    parser.add_argument("--min_negatives", type=int, default=7,
                        help="drop records with fewer surviving negatives (trainer sample size)")
    parser.add_argument("--dedup_threshold", type=float, default=0.7,
                        help="shingle Jaccard threshold, matching the data pipeline")
    parser.add_argument("--shard_size", type=int, default=100_000)
    return parser.parse_args()


# --- near-duplicate detection: identical to newscycle-gdelter/process.py ---

def shingle_set(text: str, n: int = 5) -> set:
    # zlib.crc32 rather than hash(): the latter is salted per process and
    # would make dedup non-reproducible across runs
    words = text.lower().split()
    if len(words) < n:
        return {zlib.crc32(" ".join(words).encode("utf-8"))}
    return {zlib.crc32(" ".join(words[i:i + n]).encode("utf-8"))
            for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --- IO ---

def load_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt") as f:
        for line in tqdm(f, desc="Loading records"):
            records.append(json.loads(line))
    return records


def write_shards(records, output_dir, shard_size):
    metadata = {"objective": {"self": [], "paired": [], "triplet": [["query", "document", "negatives"]]}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_start in range(0, len(records), shard_size):
        shard_num = shard_start // shard_size
        final_path = output_dir / f"shard-{shard_num:05d}.jsonl.gz"
        tmp_path = output_dir / f"shard-{shard_num:05d}.jsonl.gz.tmp"
        with gzip.open(tmp_path, "wt") as f:
            for record in tqdm(records[shard_start:shard_start + shard_size],
                               desc=f"Writing shard {shard_num:05d}"):
                record["metadata"] = metadata
                f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    return (len(records) + shard_size - 1) // shard_size


# --- embedding (gte-base, mean pooling over final hidden states) ---

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed_texts(model, tokenizer, texts, batch_size, desc):
    embeddings = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[start:start + batch_size]
            tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            pooled = mean_pooling(model(**tokenized), tokenized["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            embeddings[start:start + len(batch)] = normalized.float().cpu().numpy()
    return embeddings


# --- mining ---

def mine_group(group, records, q_emb, d_emb, k, min_negatives, dedup_threshold, stats):
    """Select hard negatives for every record of one entity. Returns
    {record_idx: [negative record_idx, ...]} for records that keep
    >= min_negatives; drops the rest."""
    sims = q_emb[group] @ d_emb[group].T  # (n, n) within-entity similarity
    shingles = {}

    def shingle_of(idx):
        if idx not in shingles:
            shingles[idx] = shingle_set(records[idx]["document"])
        return shingles[idx]

    selected_by_record = {}
    for row, i in enumerate(group):
        rec = records[i]
        month_key = (rec["year"], rec["month"])
        candidates = [
            (sims[row, col], j)
            for col, j in enumerate(group)
            if (records[j]["year"], records[j]["month"]) != month_key
            and records[j]["article_id"] != rec["article_id"]
        ]
        candidates.sort(key=lambda t: -t[0])
        stats["pool_sizes"][min(len(candidates), 80)] += 1

        pos_shingles = shingle_of(i)
        selected = []
        selected_shingles = []
        for _, j in candidates:
            cand_shingles = shingle_of(j)
            if jaccard(cand_shingles, pos_shingles) >= dedup_threshold:
                stats["screened_vs_positive"] += 1
                continue
            if any(jaccard(cand_shingles, s) >= dedup_threshold for s in selected_shingles):
                stats["screened_vs_selected"] += 1
                continue
            selected.append(j)
            selected_shingles.append(cand_shingles)
            if len(selected) == k:
                break

        if len(selected) < min_negatives:
            stats["dropped_lt_min"] += 1
            stats["dropped_entities"][rec["entity_norm"]] += 1
        else:
            selected_by_record[i] = selected
            stats["negative_counts"][len(selected)] += 1
    return selected_by_record


def main():
    args = parse_args()
    records = load_records(args.dataset)
    print(f"Loaded {len(records):,} records")

    by_entity = defaultdict(list)
    for i, rec in enumerate(records):
        by_entity[rec["entity_norm"]].append(i)
    print(f"{len(by_entity):,} entities")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512

    q_emb = embed_texts(model, tokenizer, [r["query"] for r in records], args.batch_size, "Embedding queries")
    d_emb = embed_texts(model, tokenizer, [r["document"] for r in records], args.batch_size, "Embedding documents")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    stats = {
        "screened_vs_positive": 0,
        "screened_vs_selected": 0,
        "dropped_lt_min": 0,
        "dropped_entities": Counter(),
        "negative_counts": Counter(),
        "pool_sizes": Counter(),
    }
    kept = []
    for entity in tqdm(sorted(by_entity), desc="Mining"):
        group = by_entity[entity]
        selected_by_record = mine_group(
            group, records, q_emb, d_emb, args.k, args.min_negatives, args.dedup_threshold, stats
        )
        for i in sorted(selected_by_record):
            rec = dict(records[i])
            rec["negatives"] = [records[j]["document"] for j in selected_by_record[i]]
            kept.append(rec)

    n_shards = write_shards(kept, args.output_dir, args.shard_size)

    stats_out = {
        "input": str(args.dataset),
        "params": {"k": args.k, "min_negatives": args.min_negatives,
                   "dedup_threshold": args.dedup_threshold, "model": MODEL_NAME},
        "records_in": len(records),
        "records_kept": len(kept),
        "records_dropped": stats["dropped_lt_min"],
        "drop_rate": round(stats["dropped_lt_min"] / len(records), 4),
        "screened_vs_positive": stats["screened_vs_positive"],
        "screened_vs_selected": stats["screened_vs_selected"],
        "negative_count_distribution": dict(sorted(stats["negative_counts"].items())),
        "pool_size_distribution": dict(sorted(stats["pool_sizes"].items())),
        "entities_with_drops": len(stats["dropped_entities"]),
        "top_dropped_entities": dict(stats["dropped_entities"].most_common(20)),
        "shards": n_shards,
    }
    with open(Path(args.output_dir) / "mining_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    print(json.dumps({k: v for k, v in stats_out.items()
                      if k not in ("negative_count_distribution", "pool_size_distribution",
                                   "top_dropped_entities")}, indent=2))


if __name__ == "__main__":
    main()
