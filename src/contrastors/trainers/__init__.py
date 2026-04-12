from .base import *
from .text_text import *

# Guard trainers with heavy optional deps — only text_text (encoder) is needed
try:
    from .glue import *
    _glue_available = True
except ImportError:
    _glue_available = False

try:
    from .image_text import *
    _image_text_available = True
except ImportError:
    _image_text_available = False

try:
    from .mlm import *
    _mlm_available = True
except ImportError:
    _mlm_available = False

try:
    from .mmlm import *
    _mmlm_available = True
except ImportError:
    _mmlm_available = False

try:
    from .distill import *
    _distill_available = True
except ImportError:
    _distill_available = False

TRAINER_REGISTRY = {
    "encoder": TextTextTrainer,
}
if _mlm_available:
    TRAINER_REGISTRY["mlm"] = MLMTrainer
if _mmlm_available:
    TRAINER_REGISTRY["mmlm"] = MMLMTrainer
if _glue_available:
    TRAINER_REGISTRY["glue"] = GlueTrainer
if _image_text_available:
    TRAINER_REGISTRY["clip"] = ImageTextTrainer
    TRAINER_REGISTRY["locked_text"] = ImageTextTrainer
if _distill_available:
    TRAINER_REGISTRY["distill"] = DistillTrainer
