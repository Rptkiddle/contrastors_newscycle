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
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold", "hf_subset", "languages"]

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
    
    # V2: dataset_revision is at top level of result dict
    revision = res_dict.get('dataset_revision')
    
    # V2: scores are nested under "scores" key
    scores_dict = res_dict.get('scores', {})
    
    split = "test"
    if (ds_name in TRAIN_SPLIT) and ("train" in scores_dict):
        split = "train"
    elif (ds_name in VALIDATION_SPLIT) and ("validation" in scores_dict):
        split = "validation"
    elif (ds_name in DEV_SPLIT) and ("dev" in scores_dict):
        split = "dev"
    elif "test" not in scores_dict:
        logger.info(f"Skipping {ds_name} as split {split} not present.")
        continue
    
    # V2: Each split contains a LIST of score objects
    split_results = scores_dict.get(split, [])
    
    if not split_results:
        logger.info(f"Skipping {ds_name} as split {split} is empty.")
        continue
    
    # V2: Process each score object in the list (usually just one for single-language tasks)
    for score_obj in split_results:
        # V2: Languages are inside the score object
        languages = score_obj.get('languages', ['eng-Latn'])
        hf_subset = score_obj.get('hf_subset', 'default')
        
        # Determine language for naming
        lang = languages[0] if languages else 'eng-Latn'
        
        mteb_name = f"MTEB {ds_name}"
        # Add language suffix if multilingual
        mteb_name += f" ({lang})" if len(metadata.eval_langs) > 1 else ""
        
        META_STRING += "\n" + ONE_TASK.format(
            mteb_type, 
            hf_hub_name, 
            mteb_name, 
            lang if len(metadata.eval_langs) > 1 else "default", 
            split, 
            revision
        )
        
        # V2: Scores are directly in the score object (not nested by language)
        for metric, score in score_obj.items():
            if any([x in metric for x in SKIP_KEYS]):
                continue
            
            if not isinstance(score, dict):
                # It's a simple score value
                META_STRING += "\n" + ONE_METRIC.format(
                    metric,
                    score * 100,
                )
            else:
                # It's a nested dict of scores
                for sub_metric, sub_score in score.items():
                    if any([x in sub_metric for x in SKIP_KEYS]):
                        continue
                    META_STRING += "\n" + ONE_METRIC.format(
                        f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                        sub_score * 100,
                    )

META_STRING += "\n" + MARKER
# Write metadata into the provided results folder (where the JSONs live)
output_dir = results_folder
output_path = os.path.join(output_dir, "mteb_metadata.md")
if os.path.exists(output_path):
    logger.warning(f"Overwriting {output_path}")
else:
    # results_folder should already exist, but ensure it does
    os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(META_STRING)