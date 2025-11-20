import gzip
import json
import math
import os
import re
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--query_key", default="question")
    parser.add_argument("--document_key", default="positive_ctxs")
    parser.add_argument("--negatives_key", default="hard_negative_ctxs")
    parser.add_argument("--add_title", action="store_true")
    parser.add_argument("--neighbor_multiplier", type=int, default=3,
                        help="Initial multiplier for how many neighbors to request (k * multiplier)")
    parser.add_argument("--neighbor_max", type=int, default=0,
                        help="Maximum neighbors to request (0 => no explicit cap, i.e. len(documents))")

    return parser.parse_args()


def extract_month_year(text: str):
    if not isinstance(text, str):
        return None
    m = MONTH_RE.search(text)
    y = YEAR_RE.search(text)
    if m and y:
        return f"{m.group(1)}-{y.group(0)}"
    return None


def load_dataset(path, query_key, document_key, negatives_key):
    queries = []
    documents = []
    all_data = []
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))
    else:
        files = [path]
    seen_documents = set()
    for file in tqdm(files, desc="Loading shards"):
        if file.suffix == ".gz":
            filehandler = gzip.open(file, "rt")
        else:
            filehandler = open(file, "r")
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                queries.append({query_key: data[query_key]})
                docs = data[document_key]

                # maintain documents as list of dicts {document_key: docs}
                # de-duplicate by the raw value (string or dict serialised)
                doc_key = json.dumps(docs, sort_keys=True)
                if doc_key not in seen_documents:
                    documents.append({document_key: docs})
                    seen_documents.add(doc_key)

                if negatives_key in data:
                    negatives = data[negatives_key]

                    # nq format is in list for whatever reason
                    if isinstance(negatives, str):
                        negatives = [{document_key: negatives}]
                    elif isinstance(negatives, list):
                        if len(negatives) > 0 and isinstance(negatives[0], str):
                            negatives = [{document_key: neg} for neg in negatives]
                    else:
                        raise ValueError(f"Unknown format for negatives: {negatives}")

                    # add negatives to documents if unseen
                    for neg in negatives:
                        neg_val = neg[document_key]
                        neg_key = json.dumps(neg_val, sort_keys=True)
                        if neg_key not in seen_documents:
                            documents.append({document_key: neg_val})
                            seen_documents.add(neg_key)

                all_data.append(data)

    return queries, documents, all_data


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(model, tokenizer, dataset, batch_size, key, add_title=False):
    embeddings = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Embedding {key}"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]
            if add_title:
                batch = [line.get("title", "") + " " + line[key] for line in batch]
            else:
                batch = [line[key] for line in batch]

            tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)

            model_output = model(**tokenized)
            pooled = mean_pooling(model_output, tokenized["attention_mask"])
            normalized_emb = F.normalize(pooled, p=2, dim=1)
            embeddings.extend(normalized_emb.detach().cpu().numpy())

    return embeddings


def knn_neighbors(queries, index, batch_size, k):
    all_scores, all_indices = [], []
    for i in tqdm(range(0, len(queries), batch_size), disable=dist.get_rank() != 0):
        query_embs = queries[i : i + batch_size]
        top_k_scores, top_k_indices = index.search(np.array(query_embs).astype(np.float32), k)

        all_scores.extend(top_k_scores)
        all_indices.extend(top_k_indices)

    return all_scores, all_indices


def merge_unique_preserve(old, new):
    seen = set()
    out = []
    for x in old + new:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


if __name__ == "__main__":
    dist.init_process_group(timeout=timedelta(minutes=60))
    torch.cuda.set_device(dist.get_rank())
    args = parse_args()

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    model_name = "thenlper/gte-base"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(f"cuda:{dist.get_rank()}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # load dataset and build association maps
    queries, documents, dataset = load_dataset(args.dataset, args.query_key, args.document_key, args.negatives_key)

    # precompute query dates
    q_dates = [extract_month_year(rec[args.query_key]) for rec in dataset]

    # strict mode: fail if any query is missing a month or year
    missing_idxs = [i for i, d in enumerate(q_dates) if d is None]
    if len(missing_idxs) > 0:
        if dist.get_rank() == 0:
            print(f"ERROR: {len(missing_idxs)}/{len(q_dates)} queries are missing a month or year and strict mode requires both.")
            print("Examples of queries missing date:")
            for idx in missing_idxs[:5]:
                try:
                    print(f"- {dataset[idx][args.query_key]}")
                except Exception:
                    print(f"- (could not show query at index {idx})")
        # make sure all ranks reach the barrier before exiting
        dist.barrier()
        raise SystemExit("Aborting: some queries are missing month or year; please fix the dataset input.")

    # build document index map (serialize document value for stable keying)
    doc_index_map = {}
    for idx, doc_entry in enumerate(documents):
        doc_val = doc_entry[args.document_key]
        try:
            key = json.dumps(doc_val, sort_keys=True)
        except Exception:
            # fallback to str
            key = str(doc_val)
        doc_index_map[key] = idx

    # initialize doc_dates and assign positive index mapping on each dataset record
    doc_dates = [None] * len(documents)
    for i, rec in enumerate(dataset):
        q_date = q_dates[i]
        rec_pos = rec[args.document_key]
        if isinstance(rec_pos, list):
            pos_idxs = []
            for p in rec_pos:
                try:
                    k = json.dumps(p, sort_keys=True)
                except Exception:
                    k = str(p)
                if k in doc_index_map:
                    idx = doc_index_map[k]
                    pos_idxs.append(idx)
                    # assign document date (conservative: overwrite if None)
                    if doc_dates[idx] is None:
                        doc_dates[idx] = q_date
            rec["_positive_idxs"] = pos_idxs
        else:
            try:
                k = json.dumps(rec_pos, sort_keys=True)
            except Exception:
                k = str(rec_pos)
            if k in doc_index_map:
                idx = doc_index_map[k]
                rec["_positive_idx"] = idx
                if doc_dates[idx] is None:
                    doc_dates[idx] = q_date

    # embed queries and documents
    q_embed = embed(model, tokenizer, queries, args.batch_size, args.query_key)
    d_embed = embed(model, tokenizer, documents, args.batch_size, args.document_key, add_title=args.add_title)

    del model
    torch.cuda.empty_cache()

    # create faiss index
    index = faiss.IndexFlatIP(len(q_embed[0]))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(np.array(d_embed).astype(np.float32))

    # initial neighbor count
    initial_k = min(len(documents), max(args.k * args.neighbor_multiplier, args.k))
    if args.neighbor_max and args.neighbor_max > 0:
        initial_k = min(initial_k, args.neighbor_max)

    scores, indices = knn_neighbors(q_embed, index, args.batch_size, initial_k)

    # per-query filtering with iterative escalation when needed
    for i, data in enumerate(tqdm(dataset)):
        query = data[args.query_key]
        neighbors = list(indices[i]) if i < len(indices) else []

        collected = []
        processed = set()
        ptr = 0

        # helper to check candidate validity
        def is_valid_candidate(inx):
            if inx == -1:
                return False
            # avoid positives
            if isinstance(data[args.document_key], list):
                if "_positive_idxs" in data and inx in data["_positive_idxs"]:
                    return False
            else:
                if data.get("_positive_idx", None) == inx:
                    return False
            # avoid same-time
            qd = q_dates[i]
            if qd is not None and inx < len(doc_dates) and doc_dates[inx] == qd:
                return False
            # otherwise ok
            return True

        # attempt to collect k neighbors, expanding search if needed
        current_k = len(neighbors)
        while len(collected) < args.k:
            while ptr < len(neighbors) and len(collected) < args.k:
                inx = neighbors[ptr]
                ptr += 1
                if inx in processed:
                    continue
                processed.add(inx)
                if is_valid_candidate(inx):
                    collected.append(inx)

            if len(collected) >= args.k:
                break

            # need to escalate search window
            if current_k >= len(documents):
                # exhausted corpus; stop (user requested this should not happen)
                break

            new_k = min(len(documents), max(current_k * 2, args.k))
            if args.neighbor_max and args.neighbor_max > 0:
                new_k = min(new_k, args.neighbor_max)

            # perform an expanded search for this single query
            query_vec = np.array(q_embed[i]).astype(np.float32).reshape(1, -1)
            top_scores, top_indices = index.search(query_vec, new_k)
            new_list = top_indices[0].tolist()
            neighbors = merge_unique_preserve(neighbors, new_list)
            current_k = len(neighbors)

        # assign the collected negatives (map back to document values)
        data[args.negatives_key] = [documents[idx][args.document_key] for idx in collected]

    # synchronize - ensure all ranks have completed computation before writing
    dist.barrier()

    # Only rank 0 performs shard writing to avoid concurrent writers corrupting files.
    if dist.get_rank() == 0:
        metadata = {
            "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
        }
        shard_size = 100_000
        for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
            dataset_slice = dataset[shard_start : shard_start + shard_size]
            for record in dataset_slice:
                record["metadata"] = metadata
            shard_num = shard_start // shard_size
            final_path = output_dir / f"shard-{shard_num:05d}.jsonl.gz"
            tmp_path = output_dir / f"shard-{shard_num:05d}.jsonl.gz.tmp"

            # write to temp file, flush+fsync, then atomic replace
            with gzip.open(tmp_path, "wt") as f:
                for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                    f.write(json.dumps(data) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # os.fsync may not be available or may fail on some filesystems; ignore if it fails
                    pass

            os.replace(tmp_path, final_path)
