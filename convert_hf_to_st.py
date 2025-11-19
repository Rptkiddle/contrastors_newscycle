#!/usr/bin/env python3
"""Convert a HuggingFace transformer trunk into a SentenceTransformers model and optionally push to HF Hub.

This script expects a local HF-style model folder (config.json + model weights). It will:
 - inspect the HF model config for tokenizer / seq_len / projection hints
 - create a SentenceTransformer composed of Transformer + Pooling (+ Normalize)
 - optionally attempt to add a Dense projection if you supply projection weights
 - save the SentenceTransformer locally
 - optionally push the saved ST folder to the HuggingFace Hub

Usage examples:
  # convert local HF folder and save ST locally
  python convert_hf_to_st.py --input /home/rkiddle/data/05_hf_builds/biencoder/NewsCycle_125k \
      --out /home/rkiddle/data/05_hf_builds/NewsCycle_125k_st

  # convert and push to HF (will create repo if missing, requires HF login or token)
  python convert_hf_to_st.py --input /home/rkiddle/data/05_hf_builds/biencoder/NewsCycle_125k \
      --push --repo-id rkiddle/NewsCycle_125k-st --private

If your original finetune included a projection head, supply --proj-file pointing to a pytorch state-dict
containing the projection weights (expects a linear weight/bias named like 'proj.weight' / 'proj.bias' or allow mapping).

"""
import argparse
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder


def parse_args():
    p = argparse.ArgumentParser(description="Convert HF trunk -> SentenceTransformers and optionally push to Hub")
    p.add_argument("--input", required=True, help="Path to local HF model folder (or HF repo id if --from-hub)")
    p.add_argument("--from-hub", action="store_true", help="Treat --input as a HF repo id and download it locally before conversion")
    p.add_argument("--out", default=None, help="Output folder for the SentenceTransformers model (defaults to <input>_sentence_transformers)")
    p.add_argument("--max-seq-len", type=int, default=None, help="Max seq length for the Transformer module (default from config or 512)")
    p.add_argument("--tokenizer", default=None, help="Tokenizer to use if the HF folder doesn't contain tokenizer files (default inferred from config or 'bert-base-uncased')")
    p.add_argument("--proj-file", default=None, help="Optional path to a state_dict containing projection weights to add as a Dense module")
    p.add_argument("--push", action="store_true", help="Push the saved SentenceTransformers folder to the HF Hub")
    p.add_argument("--repo-id", default=None, help="HF repo id to push to, e.g. username/repo-name (required if --push)")
    p.add_argument("--private", action="store_true", help="Create the HF repo as private when pushing")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading HF model (if it has custom code)")
    return p.parse_args()


def load_config(hf_path: Path):
    cfg_path = hf_path / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ensure_tokenizer_files(hf_path: Path, tokenizer_name: str):
    # If tokenizer files not present, download from tokenizer_name into hf_path
    from transformers import AutoTokenizer

    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt"]
    found = any((hf_path / f).exists() for f in tok_files)
    if found:
        return
    print(f"No tokenizer files found in {hf_path}. Downloading tokenizer '{tokenizer_name}' into that folder...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(hf_path)


def main():
    args = parse_args()

    # Prepare local folder
    input_path = Path(args.input)
    work_dir = None
    if args.from_hub:
        # download repo to a temp folder under ./hf_downloads/<repo-basenane>
        from huggingface_hub import snapshot_download

        print(f"Downloading {args.input} from HF hub... (this may take a while)")
        work_dir = Path(snapshot_download(repo_id=args.input, allow_patterns=["*"], local_dir_use_symlinks=False))
        print("Downloaded to", work_dir)
    else:
        if not input_path.exists():
            print(f"Input path {input_path} not found", file=sys.stderr)
            sys.exit(2)
        work_dir = input_path

    config = load_config(work_dir)
    if config is None:
        print(f"Warning: could not read config.json from {work_dir}; using defaults.")

    # Determine tokenizer and seq_len
    tokenizer_name = args.tokenizer or (config.get("tokenizer_name") if config else None) or "bert-base-uncased"
    max_seq_len = args.max_seq_len or (config.get("seq_len") if config else None) or 512
    pooling = (config.get("pooling") if config else None) or "mean"
    projection_dim = (config.get("projection_dim") if config else None)

    # Ensure tokenizer files exists in hf folder (SentenceTransformers Transformer will look for tokenizer in the same folder)
    ensure_tokenizer_files(work_dir, tokenizer_name)

    # Build SentenceTransformer
    print("Building SentenceTransformer model:")
    print(" - hf trunk folder:", work_dir)
    print(" - tokenizer:", tokenizer_name)
    print(" - max_seq_len:", max_seq_len)
    print(" - pooling:", pooling)
    print(" - projection_dim:", projection_dim)

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling, Normalize, Dense

    model_args = {"trust_remote_code": True} if args.trust_remote_code else {}
    transformer = Transformer(str(work_dir), max_seq_length=int(max_seq_len), model_args=model_args)
    emb_dim = transformer.get_word_embedding_dimension()

    if pooling != "mean":
        print(f"Note: pooling in config is '{pooling}'; this script currently implements mean-pooling. Falling back to mean.")

    pool = Pooling(emb_dim, pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    norm = Normalize()

    modules = [transformer, pool, norm]

    # Add Dense/projection if requested and available
    if args.proj_file:
        import torch

        proj_path = Path(args.proj_file)
        if not proj_path.exists():
            print(f"Projection file {proj_path} not found; skipping projection addition.")
        else:
            sd = torch.load(proj_path, map_location="cpu") if proj_path.suffix in {"", ".pt", ".pth"} else None
            # Try to detect weight & bias
            weight = None
            bias = None
            if isinstance(sd, dict):
                # common names: 'proj.weight', 'proj.bias' or 'module.proj.weight'
                for k in sd.keys():
                    if k.endswith("proj.weight") or k.endswith("proj.weight_orig") or k.endswith("proj.weight_"):
                        weight = sd[k]
                    if k.endswith("proj.bias"):
                        bias = sd[k]
                if weight is None:
                    # try some other heuristics
                    for k in sd.keys():
                        if hasattr(sd[k], 'ndim') and sd[k].ndim == 2 and sd[k].shape[1] == emb_dim:
                            weight = sd[k]
                            break

            if weight is None:
                print("Could not detect projection weights in provided file; skipping projection addition.")
            else:
                out_dim = weight.shape[0]
                dense = Dense(in_features=emb_dim, out_features=out_dim, activation_function=None, normalize_embeddings=False)
                # Load weights into Dense (Dense module has a 'linear' attribute)
                try:
                    dense.linear.weight.data.copy_(weight)
                    if bias is not None:
                        dense.linear.bias.data.copy_(bias)
                    print(f"Loaded projection weights: in_dim={emb_dim}, out_dim={out_dim}")
                    modules.append(dense)
                except Exception as e:
                    print("Failed to load projection weights into Dense module:", e)

    st = SentenceTransformer(modules=modules)

    # Output path
    out_dir = Path(args.out) if args.out else Path(str(work_dir) + "_sentence_transformers")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving SentenceTransformer to", out_dir)
    st.save(str(out_dir))

    # Optionally push to HF
    if args.push:
        if not args.repo_id:
            print("--push requested but --repo-id not provided", file=sys.stderr)
            sys.exit(3)
        repo_id = args.repo_id
        api = HfApi()
        print(f"Creating/updating HF repo {repo_id} (private={args.private})...")
        try:
            api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        except Exception as e:
            print("create_repo warning/exception - continuing if repo exists:", e)

        print("Uploading folder to HF... this may take a while")
        upload_folder(folder_path=str(out_dir), repo_id=repo_id, path_in_repo="", repo_type="model")
        print("Upload complete. Model available at: https://huggingface.co/" + repo_id)

    print("Done.")


if __name__ == "__main__":
    main()
