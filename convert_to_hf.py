from pathlib import Path
import sys
from argparse import ArgumentParser

# Make the local `src/` directory importable so this script can run without
# installing the package (e.g. `pip install -e .`). This prepends the
# repository's `src` folder to sys.path when present.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if not _SRC.exists():
    # In case the script is executed from a different layout, also try parent
    # (useful if file is moved into a subfolder temporarily).
    _SRC = _HERE.parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from contrastors.models.huggingface import NomicBertConfig, NomicBertForPreTraining, NomicVisionModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--biencoder", action="store_true")
    parser.add_argument("--vision", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.biencoder:
        config = BiEncoderConfig.from_pretrained(args.ckpt_path)
        model = BiEncoder.from_pretrained(args.ckpt_path, config=config)
        model = model.trunk
    elif args.vision:
        NomicBertConfig.register_for_auto_class()
        NomicVisionModel.register_for_auto_class("AutoModel")
        config = DualEncoderConfig.from_pretrained(args.ckpt_path)
        model = DualEncoder.from_pretrained(args.ckpt_path, config=config)
        vision = model.vision
        hf_config = NomicBertConfig(**model.vision.trunk.config.to_dict())
        model = NomicVisionModel(hf_config)

        state_dict = vision.state_dict()
        state_dict = {k.replace("trunk.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        config = NomicBertConfig.from_pretrained(args.ckpt_path)
        model = NomicBertForPreTraining.from_pretrained(args.ckpt_path, config=config)

    model.push_to_hub(args.model_name, private=args.private, use_temp_dir=False)
