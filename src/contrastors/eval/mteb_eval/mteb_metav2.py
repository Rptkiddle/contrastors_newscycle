"""
Usage: python mteb_meta.py path_to_results_folder

Creates evaluation results metadata for the model card. 
"""

import json
import logging
import os
import sys

import mteb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


results_folder = sys.argv[1].rstrip("/")
model_name = results_folder.split("/")[-1]

all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        logger.info(f"Skipping non-json {file_name}")
        continue
    # Skip model_meta.json (v2 metadata file)
    if file_name == "model_meta.json":
        logger.info(f"Skipping model metadata file {file_name}")
        continue
    with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
        results = json.load(f)
        all_results = {**all_results, **{file_name.replace(".json", ""): results}}

# Use "train" split instead
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
# Use "validation" split instead
VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MultilingualSentiment", "Ocnli"]
# Use "dev" split instead
DEV_SPLIT = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "MMarcoReranking",
    "MMarcoRetrieval",
    "MSMARCO",
    "T2Reranking",
    "T2Retrieval",
    "VideoRetrieval",
]

MARKER = "---"
TAGS = "tags:"
MTEB_TAG = "- mteb"
HEADER = "model-index:"
MODEL = f"- name: {model_name}"
RES = "  results:"

META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])


ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n      revision: {}\n    metrics:"
ONE_METRIC = "    - type: {}\n      value: {}"
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

for ds_name, res_dict in sorted(all_results.items()):
    # V2: Get task metadata using new API
    try:
        task = mteb.get_task(ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval"))
        metadata = task.metadata
    except Exception as e:
        logger.warning(f"Could not load task {ds_name}: {e}")
        continue
    
    # V2: Access metadata attributes (not dict keys)
    hf_hub_name = metadata.dataset.get('path', '')
    if "CQADupstack" in ds_name:
        hf_hub_name = "BeIR/cqadupstack"
    
    mteb_type = metadata.type
    revision = metadata.dataset.get('revision')  # Okay if it's None
    
    split = "test"
    if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
        split = "train"
    elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
        split = "validation"
    elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
        split = "dev"
    elif "test" not in res_dict:
        logger.info(f"Skipping {ds_name} as split {split} not present.")
        continue
    
    res_dict = res_dict.get(split)
    
    # V2: eval_langs now includes script (e.g., 'eng-Latn' instead of 'eng')
    eval_langs = metadata.eval_langs
    
    for lang in eval_langs:
        mteb_name = f"MTEB {ds_name}"
        mteb_name += f" ({lang})" if len(eval_langs) > 1 else ""
        
        # For English there is no language key if it's the only language
        test_result_lang = res_dict.get(lang) if len(eval_langs) > 1 else res_dict
        
        # Skip if the language was not found but it has other languages
        if test_result_lang is None:
            continue
        
        META_STRING += "\n" + ONE_TASK.format(
            mteb_type, 
            hf_hub_name, 
            mteb_name, 
            lang if len(eval_langs) > 1 else "default", 
            split, 
            revision
        )
        
        for metric, score in test_result_lang.items():
            if not isinstance(score, dict):
                score = {metric: score}
            for sub_metric, sub_score in score.items():
                if any([x in sub_metric for x in SKIP_KEYS]):
                    continue
                META_STRING += "\n" + ONE_METRIC.format(
                    f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                    # All MTEB scores are 0-1, multiply them by 100
                    sub_score * 100,
                )

META_STRING += "\n" + MARKER
if os.path.exists(f"./{model_name}/mteb_metadata.md"):
    logger.warning("Overwriting mteb_metadata.md")
elif not os.path.exists(f"./{model_name}"):
    os.mkdir(f"./{model_name}")
with open(f"./{model_name}/mteb_metadata.md", "w") as f:
    f.write(META_STRING)