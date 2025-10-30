import gzip
import json
import math
import os
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

    return parser.parse_args()


def load_dataset(path, query_key, document_key, negatives_key):
    queries = []
    documents = []
    document_months = []
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

                # parse month from the query (format guaranteed to be '...; MONTH: Xxx; ...')
                month = None
                try:
                    q = data[query_key]
                    idx = q.find('MONTH:')
                    if idx != -1:
                        # take the three-letter month token following 'MONTH:'
                        month = q[idx + len('MONTH:'):].strip()[:3]
                except Exception:
                    month = None

                if docs not in seen_documents:
                    # keep the original structure (dict with document_key) so embedding code is unchanged
                    documents.append({document_key: docs})
                    document_months.append(month)

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

                    # add negatives to corpus if unseen (we don't have month info for these)
                    for neg in negatives:
                        neg_text = neg[document_key]
                        if neg_text not in seen_documents:
                            documents.append({document_key: neg_text})
                            document_months.append(None)
                            seen_documents.add(neg_text)

                seen_documents.add(docs)

                all_data.append(data)

    return queries, documents, document_months, all_data


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
                batch = [line["title"] + " " + line[key] for line in batch]
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

    # initialize once in case we have more than one iteration
    queries, documents, document_months, dataset = load_dataset(args.dataset, args.query_key, args.document_key, args.negatives_key)

    q_embed = embed(model, tokenizer, queries, args.batch_size, args.query_key)
    d_embed = embed(model, tokenizer, documents, args.batch_size, args.document_key, add_title=args.add_title)

    del model
    torch.cuda.empty_cache()

    # OLD CODE - 
    index = faiss.IndexFlatIP(len(q_embed[0]))
    # co = faiss.GpuMultipleClonerOptions()
    # co.shard = True
    # co.useFloat16 = True
    index = faiss.index_cpu_to_all_gpus(index, ngpu=4)
    index.add(np.array(d_embed).astype(np.float32))


    scores, indices = knn_neighbors(q_embed, index, args.batch_size, args.k)

    # minimal temporal filtering: ensure negatives come from a different month than the query
    retrieval_factor = 5  # retrieve k * factor and then filter by month
    top_k = args.k * retrieval_factor
    # rerun knn if we need larger candidate set
    scores, indices = knn_neighbors(q_embed, index, args.batch_size, top_k)

    def get_doc_text(doc_entry):
        return doc_entry[args.document_key] if isinstance(doc_entry, dict) else doc_entry

    for i, data in enumerate(tqdm(dataset)):
        query = data[args.query_key]
        # extract query month (guaranteed format like '...; MONTH: Jan; ...')
        q_month = None
        idxm = query.find('MONTH:')
        if idxm != -1:
            q_month = query[idxm + len('MONTH:'):].strip()[:3]

        inxs = indices[i]
        filtered_inx = []
        for inx in inxs:
            if inx == -1:
                break
            # skip same-month documents
            doc_month = document_months[inx] if inx < len(document_months) else None
            if doc_month is not None and q_month is not None and doc_month == q_month:
                continue

            candidate_text = get_doc_text(documents[inx])

            # exclude exact positives and trivial matches
            if isinstance(data[args.document_key], list):
                positives = data.get('positive_ctxs', [])
                is_positive = False
                for p in positives:
                    p_text = p[args.document_key] if isinstance(p, dict) else p
                    if p_text == candidate_text:
                        is_positive = True
                        break
                if is_positive or candidate_text == query:
                    continue
            else:
                pos_text = data[args.document_key]
                if candidate_text == pos_text or candidate_text == query:
                    continue

            filtered_inx.append(inx)
            if len(filtered_inx) >= args.k:
                break

        # write negatives as strings
        data[args.negatives_key] = [get_doc_text(documents[inx]) for inx in filtered_inx]

        if len(data[args.negatives_key]) < args.k:
            remaining = args.k - len(data[args.negatives_key])
            kept_texts = []
            # prepare month->indices mapping once
            # build on demand for minimal code change
            month_to_indices = {}
            for idx_doc, m in enumerate(document_months):
                month_to_indices.setdefault(m, []).append(idx_doc)

            # sample from indices whose month != q_month
            allowed_indices = [idx for m, idxs in month_to_indices.items() if m != q_month for idx in idxs]
            # if allowed_indices is empty (unlikely), fall back to global random
            if not allowed_indices:
                allowed_indices = list(range(len(documents)))

            while len(kept_texts) < remaining:
                sample_idxs = np.random.choice(allowed_indices, size=remaining - len(kept_texts), replace=False).tolist()
                for sidx in sample_idxs:
                    cand = get_doc_text(documents[sidx])
                    # exclude positives/trivial
                    if isinstance(data[args.document_key], list):
                        positives = data.get('positive_ctxs', [])
                        if any((p[args.document_key] if isinstance(p, dict) else p) == cand for p in positives):
                            continue
                    else:
                        if cand == data[args.document_key] or cand == query:
                            continue
                    kept_texts.append(cand)
                    if len(kept_texts) >= remaining:
                        break

            data[args.negatives_key].extend(kept_texts)

    metadata = {
        "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
    }
    shard_size = 100_000
    for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
        dataset_slice = dataset[shard_start : shard_start + shard_size]
        for record in dataset_slice:
            record["metadata"] = metadata
        shard_num = shard_start // shard_size
        with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
            for data in tqdm(dataset_slice, desc=f"Writing shard {shard_num:05d}"):
                f.write(json.dumps(data) + "\n")
