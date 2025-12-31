#!/usr/bin/env python3
"""Convert a HuggingFace transformer trunk into a SentenceTransformers model and optionally push to HF Hub.

This script expects a local HF-style model folder (config.json + model weights + tokenizer files).

It will NOT default to any parameters - all required information must be present in the HF trunk
or explicitly provided via CLI arguments.

Required in HF trunk:
  - config.json with max_position_embeddings (or n_positions/max_seq_length) and hidden_size
  - Tokenizer files (tokenizer.json, tokenizer_config.json, vocab.txt, etc.)
  - Model weights (model.safetensors or pytorch_model.bin)

Required CLI arguments:
  - --pooling: Pooling mode (mean, cls, max, mean_sqrt_len, weightedmean, lasttoken)
  - --normalize / --no-normalize: Whether to add normalization layer

Usage examples:
  python convert_hf_to_st.py --input /path/to/hf_trunk --out /path/to/st_output --pooling mean --normalize

  python convert_hf_to_st.py --input /path/to/hf_trunk --out /path/to/st_output --pooling mean --normalize \\
      --push --repo-id username/model-name --private
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Convert HF trunk -> SentenceTransformers and optionally push to Hub")
    p.add_argument("--input", required=True, help="Path to local HF model folder (or HF repo id if --from-hub)")
    p.add_argument("--from-hub", action="store_true", help="Treat --input as a HF repo id and download it locally before conversion")
    p.add_argument("--out", default=None, help="Output folder for the SentenceTransformers model (defaults to <input>_sentence_transformers)")
    
    # Required architectural decisions (cannot be inferred from HF trunk)
    p.add_argument("--pooling", required=True, choices=["mean", "cls", "max", "mean_sqrt_len", "weightedmean", "lasttoken"],
                   help="Pooling mode for combining token embeddings into sentence embedding")
    normalize_group = p.add_mutually_exclusive_group(required=True)
    normalize_group.add_argument("--normalize", action="store_true", dest="normalize", help="Add normalization layer at the end")
    normalize_group.add_argument("--no-normalize", action="store_false", dest="normalize", help="Do not add normalization layer")
    
    # Optional overrides (will be read from config if not provided)
    p.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length (default: read from HF config)")
    
    # Hub options
    p.add_argument("--push", action="store_true", help="Push the saved SentenceTransformers folder to the HF Hub")
    p.add_argument("--repo-id", default=None, help="HF repo id to push to, e.g. username/repo-name (required if --push)")
    p.add_argument("--private", action="store_true", help="Create the HF repo as private when pushing")
    
    # Model loading options
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading HF model (if it has custom code)")
    
    return p.parse_args()


def load_config(hf_path: Path) -> dict:
    """Load and validate HF config.json. Raises if missing or invalid."""
    cfg_path = hf_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {hf_path}")
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def get_max_seq_length(config: dict) -> int:
    """Extract max sequence length from HF config. Raises if not found."""
    # Try common field names in order of preference
    for field in ["max_position_embeddings", "n_positions", "max_seq_length", "seq_len"]:
        if field in config:
            return int(config[field])
    
    raise ValueError(
        f"Could not find max sequence length in config.json. "
        f"Expected one of: max_position_embeddings, n_positions, max_seq_length, seq_len. "
        f"Available keys: {list(config.keys())}"
    )


def get_hidden_size(config: dict) -> int:
    """Extract hidden size / embedding dimension from HF config. Raises if not found."""
    for field in ["hidden_size", "dim", "d_model", "n_embd"]:
        if field in config:
            return int(config[field])
    
    raise ValueError(
        f"Could not find hidden size in config.json. "
        f"Expected one of: hidden_size, dim, d_model, n_embd. "
        f"Available keys: {list(config.keys())}"
    )


def validate_tokenizer_files(hf_path: Path):
    """Check that tokenizer files exist. Raises if missing."""
    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt"]
    found = [f for f in tok_files if (hf_path / f).exists()]
    
    if not found:
        raise FileNotFoundError(
            f"No tokenizer files found in {hf_path}. "
            f"Expected at least one of: {tok_files}"
        )
    
    return found


def validate_model_weights(hf_path: Path):
    """Check that model weights exist. Raises if missing."""
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    found = [f for f in weight_files if (hf_path / f).exists()]
    
    if not found:
        raise FileNotFoundError(
            f"No model weight files found in {hf_path}. "
            f"Expected at least one of: {weight_files}"
        )
    
    return found


def main():
    args = parse_args()

    # Prepare local folder
    input_path = Path(args.input)
    work_dir = None
    
    if args.from_hub:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.input} from HF hub... (this may take a while)")
        work_dir = Path(snapshot_download(repo_id=args.input, allow_patterns=["*"], local_dir_use_symlinks=False))
        print("Downloaded to", work_dir)
    else:
        if not input_path.exists():
            print(f"Error: Input path {input_path} not found", file=sys.stderr)
            sys.exit(2)
        work_dir = input_path

    # Validate HF trunk contents
    print(f"Validating HF trunk at {work_dir}...")
    
    config = load_config(work_dir)
    print(f"  - Found config.json")
    
    tokenizer_files = validate_tokenizer_files(work_dir)
    print(f"  - Found tokenizer files: {tokenizer_files}")
    
    weight_files = validate_model_weights(work_dir)
    print(f"  - Found weight files: {weight_files}")
    
    # Extract required parameters from config
    max_seq_length = args.max_seq_len if args.max_seq_len else get_max_seq_length(config)
    hidden_size = get_hidden_size(config)
    
    print(f"\nModel parameters:")
    print(f"  - max_seq_length: {max_seq_length}" + (" (from --max-seq-len)" if args.max_seq_len else " (from config)"))
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - pooling: {args.pooling}")
    print(f"  - normalize: {args.normalize}")

    # Build SentenceTransformer
    print("\nBuilding SentenceTransformer model...")
    
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling, Normalize

    model_args = {"trust_remote_code": True} if args.trust_remote_code else {}
    transformer = Transformer(str(work_dir), max_seq_length=max_seq_length, model_args=model_args)
    
    # Verify embedding dimension matches config
    emb_dim = transformer.get_word_embedding_dimension()
    if emb_dim != hidden_size:
        print(f"Warning: Transformer embedding dim ({emb_dim}) differs from config hidden_size ({hidden_size})")

    # Create pooling layer based on --pooling argument
    pooling_kwargs = {
        "word_embedding_dimension": emb_dim,
        "pooling_mode_cls_token": args.pooling == "cls",
        "pooling_mode_mean_tokens": args.pooling == "mean",
        "pooling_mode_max_tokens": args.pooling == "max",
        "pooling_mode_mean_sqrt_len_tokens": args.pooling == "mean_sqrt_len",
        "pooling_mode_weightedmean_tokens": args.pooling == "weightedmean",
        "pooling_mode_lasttoken": args.pooling == "lasttoken",
    }
    pool = Pooling(**pooling_kwargs)

    modules = [transformer, pool]
    
    if args.normalize:
        modules.append(Normalize())

    st = SentenceTransformer(modules=modules)

    # Output path
    out_dir = Path(args.out) if args.out else Path(str(work_dir) + "_sentence_transformers")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving SentenceTransformer to {out_dir}...")
    st.save(str(out_dir))

    # Optionally push to HF
    if args.push:
        if not args.repo_id:
            print("Error: --push requested but --repo-id not provided", file=sys.stderr)
            sys.exit(3)
        
        from huggingface_hub import HfApi, upload_folder
        
        repo_id = args.repo_id
        api = HfApi()
        print(f"\nCreating/updating HF repo {repo_id} (private={args.private})...")
        try:
            api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        except Exception as e:
            print(f"create_repo warning/exception - continuing if repo exists: {e}")

        print("Uploading folder to HF... this may take a while")
        upload_folder(folder_path=str(out_dir), repo_id=repo_id, path_in_repo="", repo_type="model")
        print("Upload complete. Model available at: https://huggingface.co/" + repo_id)

    print("\nDone.")


if __name__ == "__main__":
    main()