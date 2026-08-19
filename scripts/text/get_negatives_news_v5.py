"""Mine boundary-targeted hard negatives for NewsCycle V5.

V5 keeps V2's positives exactly (input = V2 train_{split}.jsonl, one
cascade-top article per entity-month key) and redesigns the negatives to
sit on the two decision boundaries the benchmark tests:

1. Same-entity temporal negatives (12), stratified by month distance:
   quotas NEAR (|d|<=2): 6, MID (3<=|d|<=6): 3, FAR (|d|>=7): 3,
   hardest-by-gte-cosine within each band, backfilled outward when a
   band is thin. (V4 diagnostic: unstratified cosine mining yields only
   4.8% of negatives at +-1 month — the month boundary is barely
   trained.)
2. Cross-entity same-month negatives (8): hardest-by-cosine articles
   from the query's own month whose entity list does not contain the
   query entity. Time-confusable by construction, so the model must
   separate them by entity identity — hard signal for the full-pool
   view that in-batch negatives provide only weakly.

Unchanged from the validated V3/V4 machinery: story-level Jaccard
screens at --dedup_threshold (vs positive and vs already-selected),
split safety (temporal candidates restricted to the pairs file's
months; same-month is trivially in-window), key-id false-negative
masking fields, min_negatives drop rule. Every stored negative carries
a band tag in `negative_bands` (near/mid/far/cross), so ablations are
shard filters rather than re-mines. Mask key for a cross-entity
negative: the key whose cascade-top positive it is (if any), else
(entities[0], its own month).

Run on a single GPU:
    python get_negatives_news_v5.py --pairs train_inter.jsonl \
        --pool salient_pool.jsonl.gz --output_dir out/ \
        --pool_emb_cache pool_emb.npy
"""

import argparse
import gzip
import json
import os
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_NAME = "thenlper/gte-base"

QUOTAS = {"near": 6, "mid": 3, "far": 3}   # same-entity temporal, by band
CROSS_QUOTA = 8                             # cross-entity same-month
BAND_ORDER = ["near", "mid", "far"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                        help="V2 train_{split}.jsonl[.gz] (single positive per key)")
    parser.add_argument("--pool", required=True, help="salient_pool.jsonl.gz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_negatives", type=int, default=7)
    parser.add_argument("--dedup_threshold", type=float, default=0.6)
    parser.add_argument("--near_dedup_threshold", type=float, default=None,
                        help="optional tighter screen applied only in the near band")
    parser.add_argument("--shard_size", type=int, default=100_000)
    parser.add_argument("--pool_emb_cache", default=None)
    parser.add_argument("--query_emb_cache", default=None,
                        help="npy path for query embeddings; if it exists (with the "
                             "pool cache) the encoder is never loaded")
    return parser.parse_args()


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


def compute_embeddings(query_texts, pool_texts, batch_size):
    """Load gte-base and embed whichever of the two text lists is not None.
    torch/transformers are imported here so cached runs need only numpy."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def embed_texts(model, tokenizer, texts, desc):
        embeddings = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc=desc):
                batch = texts[start:start + batch_size]
                tokenized = tokenizer(batch, padding=True, truncation=True,
                                      return_tensors="pt").to(model.device)
                pooled = mean_pooling(model(**tokenized), tokenized["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                embeddings[start:start + len(batch)] = normalized.half().cpu().numpy()
        return embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512
    q = embed_texts(model, tokenizer, query_texts, "Embedding queries") \
        if query_texts is not None else None
    p = embed_texts(model, tokenizer, pool_texts, "Embedding pool") \
        if pool_texts is not None else None
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return q, p


def main():
    args = parse_args()
    pool = load_jsonl(args.pool, "Loading pool")
    records = load_jsonl(args.pairs, "Loading pairs")
    print(f"Pool: {len(pool):,} articles   Pairs: {len(records):,} records")

    aid_to_pool = {p["article_id"]: idx for idx, p in enumerate(pool)}
    pool_by_entity = defaultdict(list)
    pool_by_month = defaultdict(list)
    for idx, p in enumerate(pool):
        for e in p["entities"]:
            pool_by_entity[e].append(idx)
        pool_by_month[(p["year"], p["month"])].append(idx)

    # join positive text from pool; verify against inline document if present
    checked_inline = 0
    for rec in records:
        pidx = aid_to_pool.get(rec["article_id"])
        if pidx is None:
            raise RuntimeError(f"pair article_id {rec['article_id']} missing from pool")
        rec["_pos_pool_idx"] = pidx
        if "document" in rec and checked_inline < 500:
            assert rec["document"] == pool[pidx]["text"], \
                f"inline/pool text mismatch for {rec['article_id']}"
            checked_inline += 1
        rec["key_id"] = key_id_for(rec["year"], rec["month"], rec["entity_norm"])
    print(f"inline-vs-pool text spot-checks passed: {checked_inline}")

    # cascade-top positive lookup for cross-entity mask keys
    aid_to_poskey = {rec["article_id"]: rec["key_id"] for rec in records}

    # split safety window from the pairs file
    allowed_months = {(r["year"], r["month"]) for r in records}
    print(f"train-month window: {len(allowed_months)} months")

    q_emb = p_emb = None
    if args.query_emb_cache and os.path.exists(args.query_emb_cache):
        q_emb = np.load(args.query_emb_cache)
        assert q_emb.shape[0] == len(records), "query embedding cache does not match pairs"
        print(f"loaded query embeddings from {args.query_emb_cache}")
    if args.pool_emb_cache and os.path.exists(args.pool_emb_cache):
        p_emb = np.load(args.pool_emb_cache)
        assert p_emb.shape[0] == len(pool), "pool embedding cache does not match pool"
        print(f"loaded pool embeddings from {args.pool_emb_cache}")
    if q_emb is None or p_emb is None:
        q_new, p_new = compute_embeddings(
            [r["query"] for r in records] if q_emb is None else None,
            [p["text"] for p in pool] if p_emb is None else None,
            args.batch_size)
        if q_emb is None:
            q_emb = q_new
            if args.query_emb_cache:
                np.save(args.query_emb_cache, q_emb)
        if p_emb is None:
            p_emb = p_new
            if args.pool_emb_cache:
                np.save(args.pool_emb_cache, p_emb)

    # per-month candidate structures, built lazily, kept for reuse
    month_cols = {}      # month -> (cand_idxs list, fp32 emb matrix, entity sets)
    def month_block(mk):
        if mk not in month_cols:
            idxs = pool_by_month.get(mk, [])
            month_cols[mk] = (idxs,
                              p_emb[idxs].astype(np.float32) if idxs else None,
                              [set(pool[i]["entities"]) for i in idxs])
        return month_cols[mk]

    stats = Counter()
    delta_hist = Counter()
    band_counts = Counter()
    dropped_entities = Counter()
    kept = []
    rescued = 0

    by_entity = defaultdict(list)
    for i, rec in enumerate(records):
        by_entity[rec["entity_norm"]].append(i)
    print(f"{len(by_entity):,} entities")

    for entity in tqdm(sorted(by_entity), desc="Mining"):
        rec_idxs = by_entity[entity]
        ent_pool = [i for i in pool_by_entity.get(entity, [])
                    if (pool[i]["year"], pool[i]["month"]) in allowed_months]
        ent_sims = None
        if ent_pool:
            ent_sims = (q_emb[rec_idxs].astype(np.float32)
                        @ p_emb[ent_pool].astype(np.float32).T)
        shingle_cache = {}   # scoped to this entity's group of records

        def sh(pidx):
            if pidx not in shingle_cache:
                shingle_cache[pidx] = shingle_set(pool[pidx]["text"])
            return shingle_cache[pidx]

        for row, i in enumerate(rec_idxs):
            rec = records[i]
            rec_mi = midx(rec["year"], rec["month"])
            pos_sh = shingle_set(pool[rec["_pos_pool_idx"]]["text"])
            selected, sel_sh, bands = [], [], []

            def try_take(pidx, band, thr_pos):
                cand = sh(pidx)
                if jaccard(cand, pos_sh) >= thr_pos:
                    stats["screened_vs_positive"] += 1
                    return False
                if any(jaccard(cand, s) >= args.dedup_threshold for s in sel_sh):
                    stats["screened_vs_selected"] += 1
                    return False
                selected.append(pidx)
                sel_sh.append(cand)
                bands.append(band)
                return True

            # ---- same-entity temporal, band quotas, hardest-first ----
            if ent_sims is not None:
                by_band = {b: [] for b in BAND_ORDER}
                for col in np.argsort(-ent_sims[row]):
                    pidx = ent_pool[col]
                    d = midx(pool[pidx]["year"], pool[pidx]["month"]) - rec_mi
                    if d == 0:
                        continue
                    by_band[band_of(d)].append(pidx)
                cursors = {b: 0 for b in BAND_ORDER}

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

                total_quota = sum(QUOTAS.values())
                for b in BAND_ORDER:
                    fill(b, QUOTAS[b])
                for b in BAND_ORDER:   # backfill outward, near first
                    if len(selected) >= total_quota:
                        break
                    n = fill(b, total_quota - len(selected))
                    if n:
                        stats[f"backfill_{b}"] += n
            n_temporal = len(selected)
            for pidx, band in zip(selected, bands):
                band_counts[band] += 1
                delta_hist[midx(pool[pidx]["year"], pool[pidx]["month"]) - rec_mi] += 1

            # ---- cross-entity same-month ----
            cand_idxs, cand_emb, cand_ents = month_block((rec["year"], rec["month"]))
            if cand_idxs:
                sims = cand_emb @ q_emb[i].astype(np.float32)
                taken = 0
                for col in np.argsort(-sims):
                    if taken >= CROSS_QUOTA:
                        break
                    if entity in cand_ents[col]:
                        continue
                    pidx = cand_idxs[col]
                    if try_take(pidx, "cross", args.dedup_threshold):
                        band_counts["cross"] += 1
                        taken += 1
                    else:
                        # relabel screen stats for cross
                        pass

            if len(selected) < args.min_negatives:
                stats["dropped_lt_min"] += 1
                dropped_entities[entity] += 1
                continue
            if n_temporal < args.min_negatives:
                rescued += 1

            out = {k: v for k, v in rec.items()
                   if not k.startswith("_") and k != "document"}
            out["document"] = pool[rec["_pos_pool_idx"]]["text"]
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

    n_shards = write_shards(kept, args.output_dir, args.shard_size)
    print(f"kept {len(kept):,} records ({stats['dropped_lt_min']} dropped, "
          f"{rescued} rescued by cross-entity supply) in {n_shards} shard(s)")

    stats_out = {
        "kept": len(kept),
        "dropped_lt_min": stats["dropped_lt_min"],
        "rescued_by_cross": rescued,
        "band_counts": dict(band_counts),
        "backfill": {b: stats.get(f"backfill_{b}", 0) for b in BAND_ORDER},
        "temporal_delta_hist": {str(k): v for k, v in sorted(delta_hist.items())},
        "screens": {k: v for k, v in stats.items() if k.startswith("screened")},
        "dropped_entities_top": dict(dropped_entities.most_common(25)),
        "quotas": {**QUOTAS, "cross": CROSS_QUOTA},
        "dedup_threshold": args.dedup_threshold,
        "near_dedup_threshold": args.near_dedup_threshold,
    }
    with open(Path(args.output_dir) / "mining_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print(json.dumps({k: stats_out[k] for k in
                      ["kept", "dropped_lt_min", "rescued_by_cross",
                       "band_counts", "backfill"]}, indent=2))


if __name__ == "__main__":
    main()
