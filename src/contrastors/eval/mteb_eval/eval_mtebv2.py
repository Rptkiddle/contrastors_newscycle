"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import os
import time
from argparse import ArgumentParser
from typing import Any

# ============================================================================
# V2 IMPORTS: Added ResultCache and proper typing for encoder protocol
# ============================================================================
import mteb
from mteb.cache import ResultCache
from mteb.encoder_interface import PromptType  # Import v2 types
from mteb.model_meta import ModelMeta

# Allow running this script directly from the repository (without pip installing).
# Ensure the repository `src/` directory is on sys.path so `import contrastors...` works.
import os
import sys

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from contrastors.eval.encoder import Encoder, HFEncoder, STransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

os.environ['OPENBLAS_NUM_THREADS'] = '16'

# ============================================================================
# TASK LISTS: No changes needed - these remain the same in v2
# ============================================================================
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


# ============================================================================
# V2 NEW: Create a ModelMeta wrapper that properly integrates with MTEB v2
# This is the CORRECT way to make a custom model work with MTEB v2
# ============================================================================
class V1EncoderModelMeta(ModelMeta):
    """
    ModelMeta wrapper for v1 encoders to work with MTEB v2.
    
    MTEB v2 expects models to be loaded through ModelMeta objects.
    This wrapper:
    1. Takes a pre-initialized v1 encoder
    2. Wraps it to be v2-compatible
    3. Returns it when load_model() is called
    """
    
    def __init__(self, v1_encoder, model_name: str):
        """
        Args:
            v1_encoder: Pre-initialized v1 encoder
            model_name: Name identifier for the model
        """
        self.v1_encoder = v1_encoder
        self._model_name = model_name
        
    def load_model(self, **kwargs) -> Any:
        """
        Load and return the v2-compatible wrapped encoder.
        
        Returns:
            V1toV2EncoderAdapter that wraps the v1 encoder
        """
        return V1toV2EncoderAdapter(self.v1_encoder)
    
    @property
    def name(self) -> str:
        return self._model_name


class V1toV2EncoderAdapter:
    """
    Adapter to make v1 encoders compatible with MTEB v2 EncoderProtocol.
    
    This class properly implements the v2 encoder interface by:
    1. Accepting the full v2 encode() signature
    2. Unpacking DataLoader batches to extract text
    3. Passing text to the v1 encoder
    
    The key difference from our previous wrapper: This is designed to work
    with MTEB's internal type checking and protocol requirements.
    """
    
    def __init__(self, v1_encoder):
        """
        Args:
            v1_encoder: An encoder with v1 signature encode(sentences: list[str])
        """
        self.v1_encoder = v1_encoder
        
        # Pass through attributes for compatibility
        for attr in ['model_name', 'max_seq_length', 'device']:
            if hasattr(v1_encoder, attr):
                setattr(self, attr, getattr(v1_encoder, attr))
    
    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: PromptType | None = None,
        **kwargs
    ):
        """
        V2 encode method with full signature matching EncoderProtocol.
        
        Args:
            inputs: DataLoader yielding batches with {"text": list[str], ...}
            task_metadata: Metadata about the task (v2 only)
            hf_split: Split being evaluated, e.g., "test", "dev" (v2 only)
            hf_subset: Subset being evaluated (v2 only)
            prompt_type: Prompt type like "query" or "passage" (v2 only)
            **kwargs: Additional arguments for encoding
        
        Returns:
            Array of embeddings
        """
        # Unpack DataLoader to extract all text sentences
        sentences = []
        for batch in inputs:
            if "text" in batch:
                sentences.extend(batch["text"])
        
        # Call v1 encoder with unpacked sentences
        # V2 metadata parameters are ignored by v1 encoder but available if needed
        embeddings = self.v1_encoder.encode(sentences, **kwargs)
        
        return embeddings
    
    def __getattr__(self, name):
        """Pass through attribute access to wrapped encoder."""
        return getattr(self.v1_encoder, name)


def parse_args():
    """Parse command line arguments - no changes needed for v2"""
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
    
    # ============================================================================
    # MODEL INITIALIZATION: No changes to this section
    # We still create the model the same way as in v1
    # ============================================================================
    if args.hf_model:
        v1_encoder = HFEncoder(args.model_name, seq_length=args.seq_length)
    else:
        v1_encoder = Encoder(
            model_name, seq_length=seq_length, tokenizer_name=tokenizer_name, matryoshka_dim=args.matryoshka_dim
        )
    print(f"Add prefix: {args.add_prefix}")
    v1_encoder = STransformer(v1_encoder, add_prefix=args.add_prefix, binarize=args.binarize)

    # ============================================================================
    # V2: Wrap the v1 encoder using ModelMeta pattern
    # This is the proper v2 way to integrate custom models
    # ============================================================================
    model_meta = V1EncoderModelMeta(v1_encoder, model_name=model_name)
    model = model_meta.load_model()
    logger.info("Wrapped v1 encoder in v2-compatible adapter")

    # ============================================================================
    # TASK PREFIX MAPPING: No changes - preserved as-is (even though not used in loop)
    # ============================================================================
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

    # ============================================================================
    # V2: Setup ResultCache for managing results
    # This replaces the simple output_folder approach from v1
    # ============================================================================
    # Build cache path name (similar to v1's output_name pattern)
    cache_path = f"results/{model_name}binarize_{args.binarize}"
    if args.matryoshka_dim:
        cache_path += f"_matryoshka_{args.matryoshka_dim}"
    
    # Create ResultCache object - this manages all result storage in v2
    cache = ResultCache(cache_path=cache_path)
    logger.info(f"Results will be saved to: {cache_path}")

    # ============================================================================
    # EVALUATION LOOP: Significant v2 changes here
    # ============================================================================
    start = time.time()
    for task_name in TASK_LIST:  # Renamed from 'task' to 'task_name' for clarity
        logger.info(f"Running task: {task_name}")
        
        # ========================================================================
        # V2 CHANGE: Split selection now happens when GETTING the task
        # Previously in v1: eval_splits was passed to evaluation.run()
        # Now in v2: eval_splits is passed to mteb.get_task()
        # ========================================================================
        eval_splits = ["dev"] if task_name == "MSMARCO" else ["test"]
        
        # V2: Get the task object with splits configured
        # This creates a properly configured task object that knows which splits to evaluate
        task = mteb.get_task(task_name, eval_splits=eval_splits)
        logger.info(f"Configured task '{task_name}' with splits: {eval_splits}")
        
        # ========================================================================
        # V1 CODE (for reference - DO NOT USE):
        # evaluation = MTEB(tasks=[task_name], task_langs=["en"])
        # evaluation.run(
        #     model, 
        #     output_folder=output_name, 
        #     eval_splits=eval_splits, 
        #     show_progress_bar=True, 
        #     batch_size=16384
        # )
        # ========================================================================
        
        # ========================================================================
        # V2 CODE: Use mteb.evaluate() with proper v2 parameters
        # ========================================================================
        results = mteb.evaluate(
            model,                          # V2: First argument is the model
            tasks=[task],                   # V2: Pass the configured task object (not just name)
            cache=cache,                    # V2: Use ResultCache for result management
            encode_kwargs={"batch_size": 16384},  # V2: batch_size goes in encode_kwargs!
            # V2 REMOVED: task_langs - language filtering done in get_task() if needed
            # V2 REMOVED: eval_splits - configured in get_task() above
            # V2 REMOVED: output_folder - replaced by cache parameter
            # V2 REMOVED: show_progress_bar - likely enabled by default
            # V2 OPTIONAL (not using): prediction_folder="path" for saving predictions
            # V2 OPTIONAL (not using): co2_tracker=True for carbon tracking
        )
        
        logger.info(f"Completed task: {task_name}")
        
        # ========================================================================
        # V2 NOTE: Results are automatically saved to the cache
        # You can access them via: results.to_dataframe() if needed
        # ========================================================================
        
        # ========================================================================
        # PRESERVED: Commented-out functionality from v1 (keeping for reference)
        # ========================================================================
        # model.doc_as_query = task_name == "QuoraRetrieval"

        # prefixes = task2prefix[task_name]
        # model.query_prefix = prefixes["query"]
        # model.docoment_prefix = prefixes["document"]
        # if task_name in TASK_LIST_CLASSIFICATION and args.no_normalize_classification is False:
        #     print("Setting normalize to False")
        #     model.set_normalize(False)
        # else:
        #     model.set_normalize(True)
        
        # Preserved: breakpoint for debugging
        breakpoint()

    end = time.time()
    print(f"Time taken (mins): {(end-start)/60}")
    
    # ============================================================================
    # V2 BONUS: You can now easily load and analyze results
    # ============================================================================
    logger.info(f"\nResults saved to cache at: {cache_path}")
    logger.info("To load results later, use:")
    logger.info(f"  cache = ResultCache(cache_path='{cache_path}')")
    logger.info(f"  results = cache.load_results(models=['{model_name}'], tasks=TASK_LIST)")
    logger.info("  df = results.to_dataframe()")