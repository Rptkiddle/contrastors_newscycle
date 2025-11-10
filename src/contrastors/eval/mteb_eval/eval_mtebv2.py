"""eval_mtebv2.py — MTEB v2-compatible evaluation script

Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores.
This file mirrors `eval_mteb.py` but uses the newer `mteb.get_benchmark` / `mteb.evaluate` API.
"""

import logging
import os
import time
from argparse import ArgumentParser

# Allow running this script directly from the repository (without pip installing).
# Ensure the repository `src/` directory is on sys.path so `import contrastors...` works.
import os
import sys

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

import mteb

from contrastors.eval.encoder import Encoder, HFEncoder, STransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

os.environ['OPENBLAS_NUM_THREADS'] = '16'

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
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
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = (
    # TASK_LIST_CLASSIFICATION
    # + TASK_LIST_CLUSTERING
    # + TASK_LIST_PAIR_CLASSIFICATION
    # + TASK_LIST_RERANKING
    # + TASK_LIST_RETRIEVAL
    # + TASK_LIST_STS
    TASK_LIST_RETRIEVAL
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--no_normalize_classification", action="store_false")
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--matryoshka_dim", type=int)
    parser.add_argument("--hf_model", action="store_true")
    parser.add_argument("--query_prefix", type=str, default="search_query: ")
    parser.add_argument("--document_prefix", type=str, default="search_document: ")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    seq_length = args.seq_length
    no_normalize_classification = args.no_normalize_classification
    if args.hf_model:
        model = HFEncoder(args.model_name, seq_length=args.seq_length)
    else:
        model = Encoder(
            model_name, seq_length=seq_length, tokenizer_name=tokenizer_name, matryoshka_dim=args.matryoshka_dim
        )
    print(f"Add prefix: {args.add_prefix}")
    model = STransformer(model, add_prefix=args.add_prefix, binarize=args.binarize)

    task2prefix = {}
    for task in TASK_LIST_CLASSIFICATION:
        task2prefix[task] = {"query": "classification", "document": "classification"}

    for task in TASK_LIST_CLUSTERING:
        task2prefix[task] = {"query": "clustering", "document": "clustering"}

    for task in TASK_LIST_PAIR_CLASSIFICATION:
        task2prefix[task] = {"query": "classification", "document": "classification"}

    for task in TASK_LIST_RERANKING:
        task2prefix[task] = {"query": "classification", "document": "classification"}

    for task in TASK_LIST_RETRIEVAL:
        task2prefix[task] = {"query": args.query_prefix, "document": args.document_prefix}

    for task in TASK_LIST_STS:
        task2prefix[task] = {"query": "classification", "document": "classification"}

    start = time.time()
    # Create the full English benchmark once (MTEB v2 style)
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")

    for task in TASK_LIST:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]

        output_name = f"results/{model_name}binarize_{args.binarize}"
        if args.matryoshka_dim:
            output_name += f"_matryoshka_{args.matryoshka_dim}"

        # The newer mteb API exposes `mteb.evaluate(model, benchmark, ...)`.
        # We pass through the same options used previously (eval_splits, output folder, batch size,
        # and language restriction). We also allow the evaluate call to accept a `tasks` filter.
        try:
            # Preferred new API call
            results = mteb.evaluate(
                model,
                benchmark,
                tasks=[task],
                task_langs=["en"],
                output_folder=output_name,
                eval_splits=eval_splits,
                show_progress_bar=True,
                batch_size=16384,
            )
        except TypeError:
            # Some mteb releases may use a slightly different signature; try without explicit task_langs
            results = mteb.evaluate(
                model,
                benchmark,
                tasks=[task],
                output_folder=output_name,
                eval_splits=eval_splits,
                show_progress_bar=True,
                batch_size=16384,
            )

        # Write a small log of results (mteb.evaluate returns a dict-like results object)
        logger.info(f"Finished task {task}; results keys: {list(results.keys()) if hasattr(results, 'keys') else type(results)}")

    end = time.time()
    print(f"Time taken (mins): {(end-start)/60}")
