"""Mine hard negatives for multi-positive NewsCycle training data (V3).

Differences from get_negatives_news.py (the single-positive V2 miner):

1. The candidate pool is the full salient pool (salient_pool.jsonl.gz from
   newscycle-gdelter process.py), not just the chosen positives — every
   deduped salient article of every training entity is a candidate.
2. Training pairs arrive without document text (train_*_mpN.jsonl.gz holds
   query + article_id + key_id); positives are joined from the pool here.
3. Every output record carries its own key_id and a negative_key_ids list
   aligned with negatives, where a negative's key is the (query entity,
   negative's month-year) pair. The trainer uses these to mask false
   negatives in the loss: a stored negative may be the correct answer for
   another query of the same entity in the batch.

Constraints and screens are otherwise identical to V2: candidates share the
query's entity with a different month-year, near-duplicates of the positive
and of already-selected negatives are excluded (5-gram crc32 shingle
Jaccard), and records with fewer than --min_negatives survivors are dropped.

Run on a single GPU:
    python get_negatives_news_mp.py --pairs train_merged_mp5.jsonl.gz \
        --pool salient_pool.jsonl.gz --output_dir out/
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
    parser.add_argument("--pairs", required=True, help="train_*_mpN.jsonl.gz (query + article_id + key_id)")
    parser.add_argument("--pool", required=True, help="salient_pool.jsonl.gz (mining pool + document texts)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20, help="negatives stored per record")
    parser.add_argument("--min_negatives", type=int, default=7,
                        help="drop records with fewer surviving negatives (trainer sample size)")
    parser.add_argument("--dedup_threshold", type=float, default=0.7,
                        help="shingle Jaccard threshold, matching the data pipeline")
    parser.add_argument("--shard_size", type=int, default=100_000)
    parser.add_argument("--pool_emb_cache", default=None,
                        help="npy path to reuse pool embeddings across runs on the same pool")
    return parser.parse_args()


# --- near-duplicate detection: identical to newscycle-gdelter/process.py ---

def shingle_set(text: str, n: int = 5) -> set:
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


def key_id_for(year: int, month: int, entity_norm: str) -> str:
    return f"{year:04d}-{month:02d}::{entity_norm}"


# --- IO ---

def load_jsonl(path, desc):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt") as f:
        for line in tqdm(f, desc=desc):
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
    embeddings = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[start:start + batch_size]
            tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            pooled = mean_pooling(model(**tokenized), tokenized["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            embeddings[start:start + len(batch)] = normalized.half().cpu().numpy()
    return embeddings


# --- mining ---

def mine_entity(rec_idxs, records, pool, pool_idxs, q_emb, p_emb, k,
                min_negatives, dedup_threshold, stats):
    """Select hard negatives for every record of one entity from the
    entity's pool articles. Returns {record_idx: [pool_idx, ...]}."""
    sims = (q_emb[rec_idxs].astype(np.float32)
            @ p_emb[pool_idxs].astype(np.float32).T)  # (n_rec, n_pool)
    shingles = {}

    def shingle_of(pidx):
        if pidx not in shingles:
            shingles[pidx] = shingle_set(pool[pidx]["text"])
        return shingles[pidx]

    pool_ym = [(pool[p]["year"], pool[p]["month"]) for p in pool_idxs]
    selected_by_record = {}
    for row, i in enumerate(rec_idxs):
        rec = records[i]
        month_key = (rec["year"], rec["month"])
        order = np.argsort(-sims[row])
        pos_shingles = shingle_set(rec["_pos_text"])
        selected = []
        selected_shingles = []
        pool_size = 0
        for col in order:
            if pool_ym[col] == month_key:
                continue
            pool_size += 1
            if len(selected) == k:
                continue
            pidx = pool_idxs[col]
            cand_shingles = shingle_of(pidx)
            if jaccard(cand_shingles, pos_shingles) >= dedup_threshold:
                stats["screened_vs_positive"] += 1
                continue
            if any(jaccard(cand_shingles, s) >= dedup_threshold for s in selected_shingles):
                stats["screened_vs_selected"] += 1
                continue
            selected.append(pidx)
            selected_shingles.append(cand_shingles)
        stats["pool_sizes"][min(pool_size, 500)] += 1

        if len(selected) < min_negatives:
            stats["dropped_lt_min"] += 1
            stats["dropped_entities"][rec["entity_norm"]] += 1
        else:
            selected_by_record[i] = selected
            stats["negative_counts"][len(selected)] += 1
            if i % 200 == 0:  # sample ~0.5% of records for the QC histogram
                for x in range(len(selected)):
                    for y in range(x + 1, len(selected)):
                        j = jaccard(shingle_of(selected[x]), shingle_of(selected[y]))
                        stats["neg_pairwise_jaccard_hist"][round(j * 20) / 20] += 1
    return selected_by_record


def main():
    args = parse_args()
    pool = load_jsonl(args.pool, "Loading pool")
    records = load_jsonl(args.pairs, "Loading pairs")
    print(f"Pool: {len(pool):,} articles   Pairs: {len(records):,} records")

    aid_to_pool = {p["article_id"]: idx for idx, p in enumerate(pool)}
    pool_by_entity = defaultdict(list)
    for idx, p in enumerate(pool):
        for e in p["entities"]:
            pool_by_entity[e].append(idx)

    # join positive texts from the pool; every pair's article must be there
    missing = 0
    for rec in records:
        pidx = aid_to_pool.get(rec["article_id"])
        if pidx is None:
            missing += 1
            rec["_pos_text"] = None
        else:
            rec["_pos_text"] = pool[pidx]["text"]
    if missing:
        raise RuntimeError(f"{missing} pair article_ids missing from pool — inconsistent inputs")

    by_entity = defaultdict(list)
    for i, rec in enumerate(records):
        by_entity[rec["entity_norm"]].append(i)
    print(f"{len(by_entity):,} entities")

    # Split safety: negatives may only come from months the split trains on.
    # The pairs file defines that window (merged spans all months, so this
    # is a no-op there; for inter/extra specialists it excludes their test
    # months, preserving the temporal train/test separation).
    allowed_months = {(r["year"], r["month"]) for r in records}
    pool_in_window = sum(1 for p in pool if (p["year"], p["month"]) in allowed_months)
    print(f"train-month window: {len(allowed_months)} months; "
          f"pool candidates in window: {pool_in_window:,}/{len(pool):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512

    q_emb = embed_texts(model, tokenizer, [r["query"] for r in records],
                        args.batch_size, "Embedding queries")
    if args.pool_emb_cache and os.path.exists(args.pool_emb_cache):
        p_emb = np.load(args.pool_emb_cache)
        assert p_emb.shape[0] == len(pool), "pool embedding cache does not match pool"
        print(f"loaded pool embeddings from {args.pool_emb_cache}")
    else:
        p_emb = embed_texts(model, tokenizer, [p["text"] for p in pool],
                            args.batch_size, "Embedding pool")
        if args.pool_emb_cache:
            np.save(args.pool_emb_cache, p_emb)
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
        # QC readout: pairwise Jaccard among each record's stored negatives
        # (sampled records), binned to 0.05 — shows the diversity the
        # selected-vs-selected screen actually achieves
        "neg_pairwise_jaccard_hist": Counter(),
    }
    kept = []
    for entity in tqdm(sorted(by_entity), desc="Mining"):
        rec_idxs = by_entity[entity]
        pool_idxs = [i for i in pool_by_entity.get(entity, [])
                     if (pool[i]["year"], pool[i]["month"]) in allowed_months]
        selected_by_record = mine_entity(
            rec_idxs, records, pool, pool_idxs, q_emb, p_emb,
            args.k, args.min_negatives, args.dedup_threshold, stats)
        for i in sorted(selected_by_record):
            rec = {k: v for k, v in records[i].items() if k != "_pos_text"}
            rec["document"] = records[i]["_pos_text"]
            rec["negatives"] = [pool[p]["text"] for p in selected_by_record[i]]
            rec["negative_key_ids"] = [
                key_id_for(pool[p]["year"], pool[p]["month"], entity)
                for p in selected_by_record[i]]
            kept.append(rec)

    n_shards = write_shards(kept, args.output_dir, args.shard_size)

    stats_out = {
        "pairs": str(args.pairs),
        "pool": str(args.pool),
        "params": {"k": args.k, "min_negatives": args.min_negatives,
                   "dedup_threshold": args.dedup_threshold, "model": MODEL_NAME},
        "pool_articles": len(pool),
        "allowed_months": len(allowed_months),
        "pool_articles_in_window": pool_in_window,
        "records_in": len(records),
        "records_kept": len(kept),
        "records_dropped": stats["dropped_lt_min"],
        "drop_rate": round(stats["dropped_lt_min"] / len(records), 4),
        "screened_vs_positive": stats["screened_vs_positive"],
        "screened_vs_selected": stats["screened_vs_selected"],
        "negative_count_distribution": dict(sorted(stats["negative_counts"].items())),
        "pool_size_distribution": dict(sorted(stats["pool_sizes"].items())),
        "neg_pairwise_jaccard_hist": dict(sorted(stats["neg_pairwise_jaccard_hist"].items())),
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
