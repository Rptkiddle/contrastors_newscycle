"""V5 hard-negative miner: boundary-stratified, deterministic (V5_PLAN section 5).

Per record (query, positive) exactly SEVEN negatives are stored and all
seven are trained on (num_negatives=7, no sampling):

  3 NEAR  same entity, |month delta| 1-2      -- hardest-by-cosine within
  1 MID   same entity, |month delta| 3-6      -- band, ranked against the
  1 FAR   same entity, |month delta| >= 7     -- record's query embedding
  2 CROSS query's own month, entity NOT in the article's entity list

Backfill when a band is thin: near -> mid -> far -> cross (cross supply
is effectively unlimited, so no record is dropped).

Machinery (V5_PLAN sections 5-7):
- Embeddings: nomic-embed-text-v1-unsupervised @2048. Documents come
  from the shared build cache (pool_emb_nomic.npy + ids); queries are
  embedded here with the "search_query: " prefix (or loaded from
  --query_emb_cache).
- Split safety: same-entity candidates restricted to months present in
  the pairs file; cross candidates are same-month (trivially in-window).
- False-negative screens: 5-gram shingle Jaccard >= --dedup_threshold
  (0.6) vs ALL of the key's selected positives and vs already-selected
  negatives. Optional tighter near-band screen via
  --near_dedup_threshold.
- Key-id masking fields: negative_key_ids aligned with negatives; a
  cross negative that is some key's positive carries THAT key, else
  (entities[0], its own month).
- Band provenance: negative_bands aligned with negatives.
- Records are SHUFFLED with a fixed seed before shard writing — the
  trainer streams shards in file order (verified: no loader-side
  shuffling), so this is where batch composition is set.
"""

import argparse
import gzip
import json
import os
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_NAME = "nomic-ai/nomic-embed-text-v1-unsupervised"
QUERY_PREFIX = "search_query: "
QUERY_MAX_TOKENS = 32
SHUFFLE_SEED = 42

QUOTAS = (("near", 3), ("mid", 1), ("far", 1), ("cross", 2))
K_TOTAL = 7


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="train_{split}_v5.jsonl.gz from build_v5_positives emit")
    ap.add_argument("--pool", required=True, help="salient_pool.jsonl.gz")
    ap.add_argument("--cache_dir", required=True,
                    help="dir with pool_emb_nomic.npy + pool_emb_nomic.ids.json.gz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256,
                    help="query embedding batch size")
    ap.add_argument("--dedup_threshold", type=float, default=0.6)
    ap.add_argument("--near_dedup_threshold", type=float, default=None)
    ap.add_argument("--shard_size", type=int, default=100_000)
    ap.add_argument("--query_emb_cache", default=None)
    return ap.parse_args()


def band_of(d):
    d = abs(d)
    if d <= 2:
        return "near"
    if d <= 6:
        return "mid"
    return "far"


def shingle_set(text, n=5):
    words = text.lower().split()
    if len(words) < n:
        return {zlib.crc32(" ".join(words).encode("utf-8"))}
    return {zlib.crc32(" ".join(words[i:i + n]).encode("utf-8"))
            for i in range(len(words) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def key_id_for(year, month, entity_norm):
    return f"{year:04d}-{month:02d}::{entity_norm}"


def midx(year, month):
    return (year - 2020) * 12 + (month - 1)


def load_jsonl(path, desc):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt") as f:
        for line in tqdm(f, desc=desc):
            records.append(json.loads(line))
    return records


def write_shards(records, output_dir, shard_size):
    metadata = {"objective": {"self": [], "paired": [],
                              "triplet": [["query", "document", "negatives"]]}}
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


def embed_queries(texts, batch_size):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      torch_dtype=torch.float16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    emb = np.empty((len(texts), 768), dtype=np.float16)
    with torch.no_grad():
        for s in tqdm(range(0, len(texts), batch_size), desc="Embedding queries"):
            batch = [QUERY_PREFIX + t for t in texts[s:s + batch_size]]
            tok = tokenizer(batch, padding=True, truncation=True,
                            max_length=QUERY_MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**tok)[0]
            mask = tok["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb[s:s + len(batch)] = F.normalize(pooled, dim=1).half().cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb


def main():
    args = parse_args()
    pool = load_jsonl(args.pool, "Loading pool")
    records = load_jsonl(args.pairs, "Loading pairs")
    print(f"Pool: {len(pool):,} articles   Pairs: {len(records):,} records")

    p_emb = np.load(Path(args.cache_dir) / "pool_emb_nomic.npy")
    ids = json.load(gzip.open(Path(args.cache_dir) / "pool_emb_nomic.ids.json.gz", "rt"))
    assert p_emb.shape[0] == len(pool) == len(ids)
    assert all(p["article_id"] == i for p, i in zip(pool, ids)), \
        "pool/embedding-cache row order mismatch"
    aid_to_pool = {a: i for i, a in enumerate(ids)}

    pool_by_entity = defaultdict(list)
    pool_by_month = defaultdict(list)
    for idx, p in enumerate(pool):
        for e in p["entities"]:
            pool_by_entity[e].append(idx)
        pool_by_month[(p["year"], p["month"])].append(idx)

    # positives of each key (for screens + cross mask keys)
    key_positives = defaultdict(list)
    aid_to_poskey = {}
    for rec in records:
        key_positives[rec["key_id"]].append(aid_to_pool[rec["article_id"]])
        aid_to_poskey[rec["article_id"]] = rec["key_id"]

    allowed_months = {(r["year"], r["month"]) for r in records}
    print(f"train-month window: {len(allowed_months)} months")

    if args.query_emb_cache and os.path.exists(args.query_emb_cache):
        q_emb = np.load(args.query_emb_cache)
        assert q_emb.shape[0] == len(records), "query cache does not match pairs"
        print(f"loaded query embeddings from {args.query_emb_cache}")
    else:
        q_emb = embed_queries([r["query"] for r in records], args.batch_size)
        if args.query_emb_cache:
            with open(args.query_emb_cache + ".tmp", "wb") as fh:
                np.save(fh, q_emb)
            os.replace(args.query_emb_cache + ".tmp", args.query_emb_cache)

    stats = Counter()
    delta_hist = Counter()
    band_counts = Counter()
    kept = []

    by_entity = defaultdict(list)
    for i, rec in enumerate(records):
        by_entity[rec["entity_norm"]].append(i)
    print(f"{len(by_entity):,} entities")

    month_block = {}
    def get_month_block(ym):
        if ym not in month_block:
            idxs = pool_by_month.get(ym, [])
            month_block[ym] = (idxs,
                               p_emb[idxs].astype(np.float32) if idxs else None,
                               [set(pool[i]["entities"]) for i in idxs])
        return month_block[ym]

    for entity in tqdm(sorted(by_entity), desc="Mining"):
        rec_idxs = by_entity[entity]
        ent_pool = [i for i in pool_by_entity.get(entity, [])
                    if (pool[i]["year"], pool[i]["month"]) in allowed_months]
        ent_sims = None
        if ent_pool:
            ent_sims = (q_emb[rec_idxs].astype(np.float32)
                        @ p_emb[ent_pool].astype(np.float32).T)
        shingle_cache = {}

        def sh(pidx):
            if pidx not in shingle_cache:
                shingle_cache[pidx] = shingle_set(pool[pidx]["text"])
            return shingle_cache[pidx]

        for row, i in enumerate(rec_idxs):
            rec = records[i]
            rec_mi = midx(rec["year"], rec["month"])
            # screen basis: ALL of this key's selected positives
            pos_idxs = key_positives[rec["key_id"]]
            pos_sh = [sh(p) for p in pos_idxs]
            pos_set = set(pos_idxs)
            selected, sel_sh, bands = [], [], []

            def try_take(pidx, band, thr_pos):
                if pidx in pos_set:
                    stats["skipped_own_positive"] += 1
                    return False
                cand = sh(pidx)
                if any(jaccard(cand, ps) >= thr_pos for ps in pos_sh):
                    stats[f"screened_vs_positive_{band}"] += 1
                    return False
                if any(jaccard(cand, s2) >= args.dedup_threshold for s2 in sel_sh):
                    stats["screened_vs_selected"] += 1
                    return False
                selected.append(pidx)
                sel_sh.append(cand)
                bands.append(band)
                return True

            # ---- same-entity temporal bands ----
            by_band = {"near": [], "mid": [], "far": []}
            if ent_sims is not None:
                for col in np.argsort(-ent_sims[row]):
                    pidx = ent_pool[col]
                    d = midx(pool[pidx]["year"], pool[pidx]["month"]) - rec_mi
                    if d == 0:
                        continue
                    by_band[band_of(d)].append(pidx)
            cursors = {b: 0 for b in by_band}

            def fill(band, want):
                got = 0
                thr = (args.near_dedup_threshold
                       if band == "near" and args.near_dedup_threshold is not None
                       else args.dedup_threshold)
                cands = by_band[band]
                while got < want and cursors[band] < len(cands):
                    pidx = cands[cursors[band]]
                    cursors[band] += 1
                    if try_take(pidx, band, thr):
                        got += 1
                return got

            temporal_quota = sum(q for b, q in QUOTAS if b != "cross")
            for b, q in QUOTAS:
                if b != "cross":
                    fill(b, q)
            for b in ("near", "mid", "far"):   # backfill outward
                if len(selected) >= temporal_quota:
                    break
                n = fill(b, temporal_quota - len(selected))
                if n:
                    stats[f"backfill_{b}"] += n
            n_temporal = len(selected)
            for pidx, band in zip(selected, bands):
                delta_hist[midx(pool[pidx]["year"], pool[pidx]["month"]) - rec_mi] += 1

            # ---- cross-entity same-month (+ any residual shortfall) ----
            cand_idxs, cand_emb, cand_ents = get_month_block((rec["year"], rec["month"]))
            want_cross = K_TOTAL - len(selected)
            if cand_idxs and want_cross > 0:
                sims = cand_emb @ q_emb[i].astype(np.float32)
                taken = 0
                for col in np.argsort(-sims):
                    if taken >= want_cross:
                        break
                    if entity in cand_ents[col]:
                        stats["cross_skipped_same_entity"] += 1
                        continue
                    if try_take(cand_idxs[col], "cross", args.dedup_threshold):
                        taken += 1
                if taken > QUOTAS[3][1]:
                    stats["cross_backfill_extra"] += taken - QUOTAS[3][1]

            if len(selected) < K_TOTAL:
                stats["dropped_lt_k"] += 1
                continue
            for b in bands:
                band_counts[b] += 1

            out = {k: v for k, v in rec.items() if k != "rank"}
            out["rank"] = rec["rank"]
            out["document"] = pool[aid_to_pool[rec["article_id"]]]["text"]
            out["negatives"] = [pool[p]["text"] for p in selected]
            out["negative_bands"] = list(bands)
            nk = []
            for pidx, band in zip(selected, bands):
                p = pool[pidx]
                if band == "cross":
                    poskey = aid_to_poskey.get(p["article_id"])
                    nk.append(poskey if poskey is not None else
                              key_id_for(p["year"], p["month"], p["entities"][0]))
                else:
                    nk.append(key_id_for(p["year"], p["month"], rec["entity_norm"]))
            out["negative_key_ids"] = nk
            kept.append(out)

    # seeded shuffle BEFORE sharding: this fixes training batch composition
    random.Random(SHUFFLE_SEED).shuffle(kept)
    n_shards = write_shards(kept, args.output_dir, args.shard_size)
    print(f"kept {len(kept):,} of {len(records):,} records "
          f"({stats['dropped_lt_k']} dropped) in {n_shards} shard(s), "
          f"shuffled with seed {SHUFFLE_SEED}")

    stats_out = {
        "kept": len(kept),
        "dropped_lt_k": stats["dropped_lt_k"],
        "band_counts": dict(band_counts),
        "backfill": {b: stats.get(f"backfill_{b}", 0) for b in ("near", "mid", "far")},
        "cross_backfill_extra": stats.get("cross_backfill_extra", 0),
        "temporal_delta_hist": {str(k): v for k, v in sorted(delta_hist.items())},
        "screens": {k: v for k, v in stats.items() if k.startswith(("screened", "skipped"))},
        "quotas": dict(QUOTAS),
        "dedup_threshold": args.dedup_threshold,
        "near_dedup_threshold": args.near_dedup_threshold,
        "shuffle_seed": SHUFFLE_SEED,
    }
    with open(Path(args.output_dir) / "mining_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print(json.dumps({k: stats_out[k] for k in
                      ["kept", "dropped_lt_k", "band_counts", "backfill",
                       "cross_backfill_extra"]}, indent=2))


if __name__ == "__main__":
    main()
