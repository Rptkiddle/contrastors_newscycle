#!/usr/bin/env python3
"""
Convert a HuggingFace-style trunk (local folder or HF repo id) into a
SentenceTransformers saved model using the repository's SentenceTransformerModule.

Example:
  python convert_hf_to_st.py \
    --hf /home/rkiddle/data/05_hf_builds/biencoder/NewsCycle_125k \
    --out /home/rkiddle/data/05_hf_builds/SentenceTransformer/NewsCycle_125k_st \
    --seq_len 2048 --pooling mean

The script will attempt to load a tokenizer from the HF folder first; if not
found it will fall back to the tokenizer name provided via --tokenizer (default
is 'bert-base-uncased' per training config).
"""
from pathlib import Path
import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert_hf_to_st")

# Make local `src` importable so we can reuse the repo's adapter class
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from contrastors.trainers.text_text import SentenceTransformerModule


def convert(hf_model_or_path: str, tokenizer_name: str, st_out_dir: str, seq_len: int, pooling: str):
    hf_model_or_path = str(hf_model_or_path)
    st_out_dir = str(st_out_dir)

    logger.info("Loading HF model trunk: %s", hf_model_or_path)
    # trust_remote_code=True is safer for custom local classes saved by this repo
    model = AutoModel.from_pretrained(hf_model_or_path, trust_remote_code=True)
    model.eval()
    # move to CPU to avoid requiring GPUs for save
    try:
        model.to("cpu")
    except Exception:
        pass

    # Try to load tokenizer from the hf folder first; fallback to provided name
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_or_path)
        logger.info("Loaded tokenizer from %s", hf_model_or_path)
    except Exception:
        logger.info("Tokenizer not found in HF folder; loading tokenizer=%s", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Wrap with repository's adapter
    adapter = SentenceTransformerModule(model=model, tokenizer=tokenizer, max_seq_length=seq_len, pooling=pooling)
    module = nn.Sequential(adapter)

    st = SentenceTransformer(modules=module, similarity_fn_name="cosine")

    out_path = Path(st_out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving SentenceTransformers model to %s", out_path)
    st.save(str(out_path))
    logger.info("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True, help="HF model id or local folder with trunk (config + weights)")
    p.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer name or folder (default: bert-base-uncased)")
    p.add_argument("--out", required=True, help="Output folder to save the SentenceTransformers model")
    p.add_argument("--seq_len", type=int, default=2048, help="Max sequence length for the tokenizer/model")
    p.add_argument("--pooling", choices=["mean", "cls", "none"], default="mean", help="Pooling to use (default: mean)")
    args = p.parse_args()

    convert(args.hf, args.tokenizer, args.out, args.seq_len, args.pooling)


if __name__ == "__main__":
    main()
