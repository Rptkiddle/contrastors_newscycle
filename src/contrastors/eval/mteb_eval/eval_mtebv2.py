"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import os
import time
from argparse import ArgumentParser

# ============================================================================
# V2 CHANGE: Import remains the same - mteb.evaluate() is available in the mteb module
# ============================================================================
import mteb  # V2: We'll use mteb.evaluate() instead of MTEB().run()

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
# V2 NEW: Wrapper class to make v1 encoders compatible with v2
# ============================================================================
class MTEBv2EncoderWrapper:
    """
    Wrapper to make v1 encoders compatible with MTEB v2.
    
    In v1, encoders had signature: encode(sentences: list[str], **kwargs)
    In v2, encoders need signature: encode(inputs: DataLoader[BatchedInput], **kwargs)
    
    This wrapper:
    1. Accepts a v1-style encoder
    2. Implements the v2 protocol
    3. Unpacks the DataLoader to extract text strings
    4. Passes them to the v1 encoder
    """
    
    def __init__(self, v1_encoder):
        """
        Args:
            v1_encoder: An encoder with v1 signature encode(sentences: list[str])
        """
        self.v1_encoder = v1_encoder
        
        # Pass through any attributes the original encoder has
        # This ensures compatibility with MTEB's internal checks
        if hasattr(v1_encoder, 'model_name'):
            self.model_name = v1_encoder.model_name
    
    def encode(self, inputs, **kwargs):
        """
        V2 encode method that accepts DataLoader[BatchedInput].
        
        Args:
            inputs: DataLoader yielding batches in format:
                    {"text": list[str], "images": list[PIL.Image], ...}
            **kwargs: Additional encoding arguments (task_name, etc.)
        
        Returns:
            Array of embeddings
        """
        # V2 CONVERSION LOGIC:
        # The DataLoader yields batches of {"text": [...], "images": [...], ...}
        # We need to unpack all text from all batches into a flat list
        sentences = []
        for batch in inputs:
            # Each batch is a dict with "text" key containing list of strings
            # V2 format: batch = {"text": ["sentence1", "sentence2", ...], ...}
            if "text" in batch:
                sentences.extend(batch["text"])
        
        # Now call the v1 encoder with the unpacked list of strings
        embeddings = self.v1_encoder.encode(sentences, **kwargs)
        
        return embeddings
    
    def __getattr__(self, name):
        """
        Pass through any other attribute access to the wrapped encoder.
        This allows the wrapper to behave like the original encoder.
        """
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
        model = HFEncoder(args.model_name, seq_length=args.seq_length)
    else:
        model = Encoder(
            model_name, seq_length=seq_length, tokenizer_name=tokenizer_name, matryoshka_dim=args.matryoshka_dim
        )
    print(f"Add prefix: {args.add_prefix}")
    model = STransformer(model, add_prefix=args.add_prefix, binarize=args.binarize)

    # ============================================================================
    # V2 NEW: Wrap the v1 encoder to make it v2-compatible
    # This allows us to use the existing encoder classes without modifying them
    # ============================================================================
    model = MTEBv2EncoderWrapper(model)
    logger.info("Wrapped model in MTEBv2EncoderWrapper for compatibility")

    # ============================================================================
    # TASK PREFIX MAPPING: No changes - preserved as-is (even though commented out in loop)
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
    # EVALUATION LOOP: Main v2 changes happen here
    # ============================================================================
    start = time.time()
    for task in TASK_LIST:
        logger.info(f"Running task: {task}")
        
        # Split selection logic remains the same
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        
        # ========================================================================
        # V1 CODE (commented out for reference):
        # evaluation = MTEB(tasks=[task], task_langs=["en"])
        # evaluation.run(
        #     model, 
        #     output_folder=output_name, 
        #     eval_splits=eval_splits, 
        #     show_progress_bar=True, 
        #     batch_size=16384
        # )
        # ========================================================================
        
        # ========================================================================
        # V2 CODE: Use mteb.evaluate() instead of MTEB().run()
        # ========================================================================
        
        # Build output folder name (same as before)
        output_name = f"results/{model_name}binarize_{args.binarize}"
        if args.matryoshka_dim:
            output_name += f"_matryoshka_{args.matryoshka_dim}"
        
        # V2: Use mteb.evaluate() - this is the new recommended approach
        mteb.evaluate(
            model,                    # V2: First argument is the model
            tasks=[task],             # V2: Second argument is tasks (was in MTEB constructor)
            task_langs=["en"],        # V2: Language filter (was in MTEB constructor)
            eval_splits=eval_splits,  # Same as v1: which splits to evaluate
            output_folder=output_name,  # Same as v1: where to save results
            batch_size=16384,         # Same as v1: batch size for encoding
            # V2 NEW OPTIONAL PARAMETERS (not using them, but available):
            # overwrite_strategy="only-missing",  # Control re-running behavior
            # cache=None,  # Use ResultCache for caching
            # prediction_folder=None,  # Save predictions for error analysis
            # encode_kwargs={},  # Additional kwargs to pass to encode()
        )
        
        # ========================================================================
        # PRESERVED: Commented-out functionality from v1 (keeping for reference)
        # ========================================================================
        # model.doc_as_query = task == "QuoraRetrieval"

        # prefixes = task2prefix[task]
        # model.query_prefix = prefixes["query"]
        # model.docoment_prefix = prefixes["document"]
        # if task in TASK_LIST_CLASSIFICATION and args.no_normalize_classification is False:
        #     print("Setting normalize to False")
        #     model.set_normalize(False)
        # else:
        #     model.set_normalize(True)
        
        # Preserved: breakpoint for debugging
        breakpoint()

    end = time.time()
    print(f"Time taken (mins): {(end-start)/60}")