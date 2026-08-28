"""V2 positive construction: coverage-proportional, cluster-then-centre.

Design of record: proj/2_NEWS/wp32-embeds/V2_PLAN.md sections 4 and 6.

Three subcommands, run in order:

  embed   Embed the salient pool with nomic-embed-text-v1-unsupervised
          at 2048 tokens ("search_document: " prefix, mean pool, L2
          norm, fp16). Shardable across GPUs: run one process per GPU
          with --shard i --nshards n, then --merge.
  select  For every key of both splits: N = min(s, round(s^0.5));
          anchor = article nearest the key centroid; k-means (k=N,
          seeded, numpy) and centroid-nearest per cluster; anchor
          replaces its own cluster's pick. Emits picks + pairwise
          pick cosines + tau calibration data + inspection samples.
          NO redundancy screen here — tau is chosen at the inspection
          checkpoint (V2_PLAN 8.1).
  emit    Apply the redundancy screen at --tau (adaptive N_eff), draw
          one fresh query template per pair (uniform over 192,
          vendored from newscycle-gdelter process.py@679fc12,
          asserted), write train_{split}_v2.jsonl.gz + build_stats.

Record order note: pairs files are NOT shuffled here; the seeded
shuffle happens at miner shard-write time (V2_PLAN section 7), because
the miner's output is what the trainer streams.
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
DOC_PREFIX = "search_document: "
MAX_TOKENS = 2048
EMB_DTYPE = np.float16
ALPHA = 0.5
KMEANS_SEED = 42
TEMPLATE_SEED = 42


# ---------------------------------------------------------------------------
# Query templates — vendored VERBATIM from newscycle-gdelter
# process.py@679fc12 (query_template_generator). Keep in sync manually;
# the 192 assertion below guards drift in count, not wording.
# ---------------------------------------------------------------------------

def query_template_generator(seed: int):
    rng = random.Random(seed)
    time_preps = ["from", "in", "for", "during"]
    rel_preps = [
        "about", "on", "involving", "mentioning",
        "related to", "concerning", "regarding",
        "featuring", "covering", "surrounding",
    ]
    date_formats = ["{MMM} {YYYY}", "{YYYY} {MMM}"]
    templates = []
    for df in date_formats:
        for tp in time_preps:
            for rp in rel_preps:
                templates.append(f"News {tp} {df} {rp} {{ENTITIES}}")
                templates.append(f"News {rp} {{ENTITIES}} {tp} {df}")
        for rp in rel_preps:
            templates.append(f"{df} news {rp} {{ENTITIES}}")
        for tp in time_preps:
            templates.append(f"{{ENTITIES}} news {tp} {df}")
        templates.append(f"{df} {{ENTITIES}} news")
        templates.append(f"{{ENTITIES}} {df} news")
    assert len(templates) == len(set(templates)) == 192
    while True:
        yield rng.choice(templates)


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


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def render_query(template: str, entity: str, year: int, month: int) -> str:
    return (template
            .replace("{MMM}", MONTH_NAMES[month - 1])
            .replace("{YYYY}", str(year))
            .replace("{ENTITIES}", entity))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_pool(path):
    pool = []
    with gzip.open(path, "rt") as f:
        for line in tqdm(f, desc="Loading pool"):
            pool.append(json.loads(line))
    return pool


def load_split_keys(path):
    """From a v2pairs file: key set, display-entity map, split months."""
    keys, display = set(), {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            k = (r["entity_norm"], r["year"], r["month"])
            keys.add(k)
            display[r["entity_norm"]] = r["entity"]
    return keys, display


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def cmd_embed(args):
    pool = load_pool(args.pool)
    n = len(pool)
    lo = args.shard * n // args.nshards
    hi = (args.shard + 1) * n // args.nshards
    out = Path(args.cache_dir) / f"pool_emb_nomic.part{args.shard:02d}.npy"
    if out.exists():
        print(f"{out} exists, skipping")
        return
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                      torch_dtype=torch.float16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = MAX_TOKENS

    texts = [DOC_PREFIX + p["text"] for p in pool[lo:hi]]
    emb = np.empty((len(texts), 768), dtype=EMB_DTYPE)
    bs = args.batch_size
    with torch.no_grad():
        for s in tqdm(range(0, len(texts), bs), desc=f"Embedding shard {args.shard}"):
            batch = texts[s:s + bs]
            tok = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**tok)[0]
            mask = tok["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb[s:s + len(batch)] = F.normalize(pooled, dim=1).half().cpu().numpy()
    # np.save appends ".npy" to bare paths; write via file handle so the
    # tmp name is exact, then atomically rename
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, emb)
    os.replace(tmp, out)
    print(f"wrote {out} ({emb.shape})")


def cmd_merge(args):
    pool = load_pool(args.pool)
    n = len(pool)
    parts = []
    for i in range(args.nshards):
        p = Path(args.cache_dir) / f"pool_emb_nomic.part{i:02d}.npy"
        parts.append(np.load(p))
    emb = np.concatenate(parts)
    assert emb.shape == (n, 768), emb.shape
    # row i of the cache corresponds to line i of the pool file; persist the
    # article_id order alongside so no consumer ever has to assume it
    ids = [p["article_id"] for p in pool]
    out = Path(args.cache_dir) / "pool_emb_nomic.npy"
    np.save(out, emb)
    with gzip.open(Path(args.cache_dir) / "pool_emb_nomic.ids.json.gz", "wt") as f:
        json.dump(ids, f)
    print(f"wrote {out} {emb.shape} + ids")
    for i in range(args.nshards):
        os.remove(Path(args.cache_dir) / f"pool_emb_nomic.part{i:02d}.npy")


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def kmeans(X, k, seed, iters=50):
    """Plain numpy k-means with k-means++ init. Deterministic via seed.
    X: (s, d) float32, L2-normalised. Returns labels (s,)."""
    rng = np.random.default_rng(seed)
    s = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(s)]
    d2 = np.full(s, np.inf, dtype=np.float32)
    for j in range(1, k):
        d2 = np.minimum(d2, ((X - centers[j - 1]) ** 2).sum(1))
        total = d2.sum()
        if total <= 0:
            centers[j:] = X[rng.integers(s, size=k - j)]
            break
        centers[j] = X[rng.choice(s, p=d2 / total)]
    labels = np.zeros(s, dtype=np.int64)
    for _ in range(iters):
        sims = X @ centers.T
        new = sims.argmax(1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c = X[m].mean(0)
                nrm = np.linalg.norm(c)
                if nrm > 0:
                    centers[j] = c / nrm
    return labels


def adaptive_n(s: int) -> int:
    return max(1, min(s, round(s ** ALPHA)))


def cmd_select(args):
    pool = load_pool(args.pool)
    emb = np.load(Path(args.cache_dir) / "pool_emb_nomic.npy")
    ids = json.load(gzip.open(Path(args.cache_dir) / "pool_emb_nomic.ids.json.gz", "rt"))
    assert len(pool) == emb.shape[0] == len(ids)
    assert all(p["article_id"] == i for p, i in zip(pool, ids)), \
        "pool/embedding-cache row order mismatch"

    by_key = defaultdict(list)   # (entity_norm, y, m) -> pool idxs
    for idx, p in enumerate(pool):
        for e in p["entities"]:
            by_key[(e, p["year"], p["month"])].append(idx)

    rng_cal = random.Random(20260820)
    for split in args.splits.split(","):
        keys, _ = load_split_keys(
            Path(args.input_dir) / f"train_{split}_v2pairs.jsonl.gz")
        out_picks = {}
        stats = Counter()
        n_hist, s_hist = Counter(), Counter()
        calib = []   # candidate same-story pairs for tau calibration
        for key in tqdm(sorted(keys), desc=f"select {split}"):
            idxs = by_key.get(key, [])
            if not idxs:
                stats["keys_missing_from_pool"] += 1
                continue
            s = len(idxs)
            N = adaptive_n(s)
            X = emb[idxs].astype(np.float32)
            centroid = X.mean(0)
            centroid /= max(np.linalg.norm(centroid), 1e-9)
            anchor_local = int((X @ centroid).argmax())
            if N == 1:
                order = [anchor_local]
            else:
                # stable per-key seed: reproducible across processes
                # (zlib.crc32, NOT hash() which is salted per process)
                key_seed = (KMEANS_SEED * 1_000_003
                            + zlib.crc32(f"{key[0]}|{key[1]}|{key[2]}".encode())) % 2**32
                labels = kmeans(X, N, key_seed)
                picks = {}
                for j in range(N):
                    m = np.where(labels == j)[0]
                    if len(m) == 0:
                        continue
                    c = X[m].mean(0); c /= max(np.linalg.norm(c), 1e-9)
                    picks[j] = (int(m[(X[m] @ c).argmax()]), len(m))
                # anchor replaces its own cluster's pick
                aj = int(labels[anchor_local])
                if aj in picks:
                    picks[aj] = (anchor_local, picks[aj][1])
                else:
                    picks[aj] = (anchor_local, 1)
                ordered = sorted(picks.values(), key=lambda t: -t[1])
                order = [anchor_local] + [i for i, _ in ordered if i != anchor_local]
            sel = [idxs[i] for i in order]
            # pairwise cosines among picks (for the emit-stage screen + calibration)
            P = emb[sel].astype(np.float32)
            cos = (P @ P.T).tolist()
            out_picks["|".join([key[0], str(key[1]), str(key[2])])] = {
                "picks": [ids[i] for i in sel], "s": s, "N": len(sel),
                "cos": [[round(c, 4) for c in row] for row in cos],
            }
            n_hist[len(sel)] += 1; s_hist[min(s, 200)] += 1
            stats["keys"] += 1; stats["pairs_before_screen"] += len(sel)
            # calibration sample: within-key pick pairs, WITH shingle
            # Jaccard so the inspection can separate wire-rewrite pairs
            # (Jaccard 0.5-0.7 zone) from distinct-story pairs (V2_PLAN 4.4)
            if len(sel) > 1 and rng_cal.random() < 0.02:
                sh = [shingle_set(pool[i]["text"]) for i in sel]
                for a in range(len(sel)):
                    for b in range(a + 1, len(sel)):
                        calib.append({
                            "a": ids[sel[a]], "b": ids[sel[b]],
                            "cos": round(cos[a][b], 4),
                            "jaccard": round(jaccard(sh[a], sh[b]), 4),
                            "key": f"{key[0]}|{key[1]}|{key[2]}"})
        od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
        with gzip.open(od / f"picks_{split}.json.gz", "wt") as f:
            json.dump(out_picks, f)
        json.dump({"stats": dict(stats), "N_hist": {str(k): v for k, v in sorted(n_hist.items())},
                   "s_hist": {str(k): v for k, v in sorted(s_hist.items())}},
                  open(od / f"select_stats_{split}.json", "w"), indent=2)
        with open(od / f"tau_calibration_{split}.jsonl", "w") as f:
            for row in calib:
                f.write(json.dumps(row) + "\n")
        print(f"{split}: {dict(stats)}")


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def cmd_emit(args):
    assert args.tau is not None, "emit requires --tau (decided at inspection)"
    pool = load_pool(args.pool)
    aid_to_pool = {p["article_id"]: i for i, p in enumerate(pool)}
    for split in args.splits.split(","):
        picks = json.load(gzip.open(Path(args.out_dir) / f"picks_{split}.json.gz", "rt"))
        _, display = load_split_keys(
            Path(args.input_dir) / f"train_{split}_v2pairs.jsonl.gz")
        gen = query_template_generator(TEMPLATE_SEED)
        out_path = Path(args.out_dir) / f"train_{split}_v2.jsonl.gz"
        stats = Counter(); neff_hist = Counter(); stub_count = 0
        with gzip.open(out_path, "wt") as f:
            for key in sorted(picks):
                ent_norm, y, m = key.split("|"); y, m = int(y), int(m)
                rec = picks[key]
                kept = []
                for i in range(rec["N"]):
                    if any(rec["cos"][i][j] > args.tau for j in kept):
                        stats["screened"] += 1
                        continue
                    kept.append(i)
                neff_hist[len(kept)] += 1
                stats["pairs"] += len(kept)
                for rank, i in enumerate(kept):
                    aid = rec["picks"][i]
                    art = pool[aid_to_pool[aid]]
                    if not art.get("text") or len(art["text"].split()) < 60:
                        stub_count += 1
                    q = render_query(next(gen), display[ent_norm], y, m)
                    f.write(json.dumps({
                        "query": q, "key_id": f"{y:04d}-{m:02d}::{ent_norm}",
                        "rank": rank, "entity": display[ent_norm],
                        "entity_norm": ent_norm, "year": y, "month": m,
                        "article_id": aid, "url": art["url"],
                    }) + "\n")
        json.dump({"tau": args.tau, "stats": dict(stats), "stub_picks_lt60w": stub_count,
                   "N_eff_hist": {str(k): v for k, v in sorted(neff_hist.items())}},
                  open(Path(args.out_dir) / f"emit_stats_{split}.json", "w"), indent=2)
        print(f"{split}: {dict(stats)} stubs<60w={stub_count} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "merge", "select", "emit"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--input_dir", help="dir with train_{split}_v2pairs.jsonl.gz")
    ap.add_argument("--out_dir")
    ap.add_argument("--splits", default="inter,extra")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()
    {"embed": cmd_embed, "merge": cmd_merge,
     "select": cmd_select, "emit": cmd_emit}[args.cmd](args)


if __name__ == "__main__":
    main()
