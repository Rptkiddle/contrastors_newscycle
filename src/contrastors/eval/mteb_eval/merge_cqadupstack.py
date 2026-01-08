"""
Merges CQADupstack subset results
Usage: python merge_cqadupstack.py path_to_results_folder
"""

import argparse
import glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_LIST_CQA = [
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
]

NOAVG_KEYS = [
    "evaluation_time",
    "mteb_version",
    "mteb_dataset_name",
    "dataset_revision",
]



parser = argparse.ArgumentParser(description="Merge CQADupstack subset results in a folder.")
parser.add_argument("results_folder", help="Path to folder containing CQADupstack JSON files")
parser.add_argument(
    "-o",
    "--output",
    help=(
        "Path to write merged JSON. If omitted, writes `CQADupstackRetrieval.json` inside the results folder."
    ),
    default=None,
)
args = parser.parse_args()
results_folder = args.results_folder.rstrip(os.sep)

# Ensure at least 1 character btw CQADupstack & Retrieval
glob_path = os.path.join(results_folder, "CQADupstack*?*Retrieval.json")
files = glob.glob(glob_path)

logger.info(f"Found CQADupstack files: {files}")

if len(files) == len(TASK_LIST_CQA):
    all_results = {}
    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as f:
            results = json.load(f)
            for split, split_results in results.items():
                if split not in ("train", "validation", "dev", "test"):
                    all_results[split] = split_results
                    continue
                all_results.setdefault(split, {})
                for metric, score in split_results.items():
                    all_results[split].setdefault(metric, 0)
                    if metric == "evaluation_time":
                        score = all_results[split][metric] + score
                    elif metric not in NOAVG_KEYS:
                        score = all_results[split][metric] + score * 1 / len(TASK_LIST_CQA)
                    all_results[split][metric] = score
    all_results["mteb_dataset_name"] = "CQADupstackRetrieval"

    # Determine output file
    if args.output:
        output_file = os.path.abspath(args.output)
    else:
        output_file = os.path.join(results_folder, "CQADupstackRetrieval.json")

    if os.path.exists(output_file):
        logger.warning(f"Overwriting {output_file}")
    else:
        out_dir = os.path.dirname(output_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    logger.info("Saving merged results to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
else:
    logger.warning(
        f"Got {len(files)}, but expected {len(TASK_LIST_CQA)} files. Missing: {set(TASK_LIST_CQA) - set([x.split('/')[-1].split('.')[0] for x in files])}; Too much: {set([x.split('/')[-1].split('.')[0] for x in files]) - set(TASK_LIST_CQA)}"
    )
