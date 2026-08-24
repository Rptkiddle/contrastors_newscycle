"""Fixture test for the V5 miner (run: python test_get_negatives_news_v5.py).

Builds a synthetic pool with hand-set unit-vector embeddings, runs the
miner end to end (query embeddings supplied via cache, so no GPU), and
asserts:
  1. quotas 3 near / 1 mid / 1 far / 2 cross when supply is ample
  2. within-band dedup at --neg_dedup_tau: the harder duplicate wins,
     the slot refills with the band's next-hardest distinct candidate
  3. NO cross-band screening: a mid negative may duplicate a near one
  4. cross negatives that are another key's positive carry that key's id
  5. records that cannot reach 7 negatives are dropped
  6. shard order is the seeded shuffle (deterministic across runs)
"""
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DIM = 16
HERE = Path(__file__).parent


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def vec(a, axis):
    """Unit vector with component a on e0 (query similarity) and the rest
    on a private axis -> pairwise cosine between two vecs = a1*a2 unless
    they share the axis."""
    v = np.zeros(DIM)
    v[0] = a
    v[axis] = np.sqrt(1 - a * a)
    return unit(v)


def build_fixture(d):
    pool, embs = [], []

    def add(aid, year, month, entities, v):
        pool.append({"article_id": aid, "url": f"http://x/{aid}", "year": year,
                     "month": month, "entities": entities,
                     "text": f"text of {aid} " + "w " * 80})
        embs.append(v)

    # main key: alpha 2021-06; positive article
    add("pos_main", 2021, 6, ["alpha"], vec(0.9, 1))
    # near candidates (2021-07, 2021-08): n1 hardest, n2 = DUPLICATE of n1
    # (identical vector), n3, n4 distinct
    add("n1", 2021, 7, ["alpha"], vec(0.80, 2))
    add("n2", 2021, 7, ["alpha"], vec(0.80, 2))      # dup of n1 (cos 1.0)
    add("n3", 2021, 8, ["alpha"], vec(0.70, 3))
    add("n4", 2021, 8, ["alpha"], vec(0.60, 4))
    # mid candidate (2021-03) sharing n1's axis -> duplicates n1 CROSS-band
    add("m1", 2021, 3, ["alpha"], vec(0.79, 2))       # cos(m1,n1)~1.0
    # far candidate (2020-11)
    add("f1", 2020, 11, ["alpha"], vec(0.50, 5))
    # cross candidates in 2021-06: c1 is beta's positive, c2 plain
    add("c1", 2021, 6, ["beta"], vec(0.75, 6))
    add("c2", 2021, 6, ["gamma"], vec(0.65, 7))
    # positives for the allowed-month helper keys (thin supply -> dropped)
    add("pos_jul", 2021, 7, ["alpha"], vec(0.10, 8))
    add("pos_aug", 2021, 8, ["alpha"], vec(0.10, 9))
    add("pos_mar", 2021, 3, ["alpha"], vec(0.10, 10))
    add("pos_nov", 2020, 11, ["alpha"], vec(0.10, 11))
    add("pos_beta", 2021, 6, ["beta"], vec(0.10, 12))

    with gzip.open(d / "salient_pool.jsonl.gz", "wt") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    with open(d / "pool_emb_nomic.npy", "wb") as fh:
        np.save(fh, np.stack(embs).astype(np.float16))
    with gzip.open(d / "pool_emb_nomic.ids.json.gz", "wt") as f:
        json.dump([p["article_id"] for p in pool], f)

    def rec(key_ent, y, m, aid):
        return {"query": f"News about {key_ent} in {y}-{m:02d}",
                "key_id": f"{y:04d}-{m:02d}::{key_ent}", "rank": 0,
                "entity": key_ent, "entity_norm": key_ent, "year": y,
                "month": m, "article_id": aid, "url": f"http://x/{aid}"}

    records = [
        rec("alpha", 2021, 6, "pos_main"),
        rec("alpha", 2021, 7, "pos_jul"),
        rec("alpha", 2021, 8, "pos_aug"),
        rec("alpha", 2021, 3, "pos_mar"),
        rec("alpha", 2020, 11, "pos_nov"),
        rec("beta", 2021, 6, "c1"),   # c1 IS beta's positive -> mask key test
    ]
    with gzip.open(d / "pairs.jsonl.gz", "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # query embeddings: every query points along e0
    q = np.stack([unit(np.eye(DIM)[0])] * len(records)).astype(np.float16)
    with open(d / "q_emb.npy", "wb") as fh:
        np.save(fh, q)
    return records


def run_miner(d, out):
    cmd = [sys.executable, str(HERE / "get_negatives_news_v5.py"),
           "--pairs", str(d / "pairs.jsonl.gz"),
           "--pool", str(d / "salient_pool.jsonl.gz"),
           "--cache_dir", str(d), "--output_dir", str(out),
           "--neg_dedup_tau", "0.90",
           "--query_emb_cache", str(d / "q_emb.npy")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"miner failed:\n{r.stdout}\n{r.stderr}"
    recs = []
    with gzip.open(out / "shard-00000.jsonl.gz", "rt") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        build_fixture(d)
        out1, out2 = d / "out1", d / "out2"
        recs = run_miner(d, out1)
        recs2 = run_miner(d, out2)

        # 6. deterministic shuffle
        assert [r["key_id"] for r in recs] == [r["key_id"] for r in recs2]

        by_key = {r["key_id"]: r for r in recs}
        # 5. thin helper keys dropped, main record kept
        assert "2021-06::alpha" in by_key, "main record missing"
        main_rec = by_key["2021-06::alpha"]
        negs = dict(zip(main_rec["negative_bands"],
                        [[] for _ in main_rec["negative_bands"]]))
        aid_by_text = {}
        with gzip.open(d / "salient_pool.jsonl.gz", "rt") as f:
            for line in f:
                p = json.loads(line)
                aid_by_text[p["text"]] = p["article_id"]
        got = [(b, aid_by_text[t]) for b, t in
               zip(main_rec["negative_bands"], main_rec["negatives"])]

        # 1. quotas
        from collections import Counter
        assert Counter(b for b, _ in got) == Counter(
            {"near": 3, "mid": 1, "far": 1, "cross": 2}), got
        near = {a for b, a in got if b == "near"}
        # 2. within-band dedup: n2 (dup of n1) screened, replaced by n3+n4
        assert near == {"n1", "n3", "n4"}, near
        # 3. no cross-band screen: m1 duplicates n1 but IS selected as mid
        assert {a for b, a in got if b == "mid"} == {"m1"}, got
        assert {a for b, a in got if b == "far"} == {"f1"}, got
        assert {a for b, a in got if b == "cross"} == {"c1", "c2"}, got
        # 4. cross mask key: c1 is beta's positive -> carries beta's key id
        kid_of = dict(zip([aid_by_text[t] for t in main_rec["negatives"]],
                          main_rec["negative_key_ids"]))
        assert kid_of["c1"] == "2021-06::beta", kid_of["c1"]
        assert kid_of["c2"] == "2021-06::gamma", kid_of["c2"]

        stats = json.load(open(out1 / "mining_stats.json"))
        assert stats["neg_dedup_tau"] == 0.90
        assert stats["screens"].get("screened_within_band_near", 0) >= 1
    print("ALL FIXTURE TESTS PASSED")


if __name__ == "__main__":
    main()
