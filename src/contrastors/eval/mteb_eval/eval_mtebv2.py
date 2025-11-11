"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import os
import time
from argparse import ArgumentParser

# ============================================================================
# V2 IMPORTS
# ============================================================================
import mteb
from mteb.cache import ResultCache
from mteb.types import PromptType  

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
# TASK LISTS: No changes needed (same tasks as v1)
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
# V2: Simple adapter implementing EncoderProtocol
# ============================================================================

from typing import Any
import numpy as np
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

class MTEBv2EncoderAdapter:
    """
    Adapter to make v1 encoders compatible with MTEB v2 EncoderProtocol.
    """
    
    def __init__(self, v1_encoder, model_name: str | None = None, revision: str | None = None, **kwargs: Any) -> None:
        """
        Initialize adapter with a v1 encoder.
        
        Args:
            v1_encoder: Encoder with v1 signature encode(sentences: list[str], **kwargs)
            model_name: Name of the model (for MTEB compatibility)
            revision: Model revision (for MTEB compatibility)
        """
        self.v1_encoder = v1_encoder
        self._model_name = model_name or getattr(v1_encoder, 'model_name', 'custom_model')
        self._revision = revision
        
        # Pass through any attributes the v1 encoder has for compatibility
        for attr in ['max_seq_length', 'device']:
            if hasattr(v1_encoder, attr):
                setattr(self, attr, getattr(v1_encoder, attr))
    
    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        """
        Encode inputs according to MTEB v2 EncoderProtocol.
        """
        # Step 1: Unpack the DataLoader to extract all text sentences
        sentences = []
        for batch in inputs:
            if "text" in batch:
                sentences.extend(batch["text"])
        
        # Step 2: Call the v1 encoder with the list of sentences
        embeddings = self.v1_encoder.encode(sentences, **kwargs)
        
        # Convert to numpy array if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def similarity(self, embeddings1, embeddings2):
        """
        Compute the similarity between two collections of embeddings.
        
        Returns a matrix of shape [num_embeddings_1, num_embeddings_2].
        """
        import torch
        
        # Convert to torch tensors if needed
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        
        # Compute cosine similarity matrix
        # Normalize embeddings
        embeddings1_norm = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2_norm = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings1_norm, embeddings2_norm.t())
        
        return similarity_matrix
    
    def similarity_pairwise(self, embeddings1, embeddings2):
        """
        Compute pairwise similarity between corresponding pairs of embeddings.
        
        Returns a vector of shape [num_embeddings].
        """
        import torch
        
        # Convert to torch tensors if needed
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        
        # Normalize embeddings
        embeddings1_norm = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2_norm = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        # Compute pairwise cosine similarity
        pairwise_similarity = torch.sum(embeddings1_norm * embeddings2_norm, dim=1)
        
        return pairwise_similarity
    
    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Return model metadata."""
        return ModelMeta(
            name=f"custom/{self._model_name}" if '/' not in self._model_name else self._model_name,
            revision=self._revision or "unknown",
            release_date=None,
            languages=None,
            loader="custom",  # Required field
            n_parameters=None,  # Can be None if unknown
            memory_usage_mb=None,  # Can be None if unknown
            max_tokens=getattr(self.v1_encoder, 'max_seq_length', 512),
            embed_dim=None,  # Will be inferred from embeddings
            license=None,
            open_weights=False,
            public_training_code=False,
            public_training_data=False,
            framework=["PyTorch"],  # Assuming PyTorch based on your encoder
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets={},
        )
    
    def __getattr__(self, name):
        """Pass through any other attribute access to the wrapped v1 encoder."""
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
    # MODEL INITIALIZATION: Same as v1
    # ============================================================================
    if args.hf_model:
        v1_encoder = HFEncoder(args.model_name, seq_length=args.seq_length)
    else:
        v1_encoder = Encoder(
            model_name, seq_length=seq_length, tokenizer_name=tokenizer_name, 
            matryoshka_dim=args.matryoshka_dim
        )
    print(f"Add prefix: {args.add_prefix}")
    v1_encoder = STransformer(v1_encoder, add_prefix=args.add_prefix, binarize=args.binarize)

    # ============================================================================
    # V2: Wrap v1 encoder with v2 adapter
    # Hopefully works - just wrap and pass to mteb.evaluate()
    # ============================================================================
    model = MTEBv2EncoderAdapter(v1_encoder)
    logger.info("Created v2-compatible encoder adapter")

    # ============================================================================
    # TASK PREFIX MAPPING: Preserved from v1 (was not actually used there)
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
    # V2: Setup ResultCache for result management
    # ============================================================================
    cache_path = f"results/{model_name}binarize_{args.binarize}"
    if args.matryoshka_dim:
        cache_path += f"_matryoshka_{args.matryoshka_dim}"
    
    cache = ResultCache(cache_path=cache_path)
    logger.info(f"Results will be saved to: {cache_path}")

    # ============================================================================
    # EVALUATION LOOP: v2 changes
    # ============================================================================
    start = time.time()
    for task_name in TASK_LIST:
        logger.info(f"Running task: {task_name}")
        
        # V2: Configure split when getting the task
        eval_splits = ["dev"] if task_name == "MSMARCO" else ["test"]
        task = mteb.get_task(task_name, eval_splits=eval_splits)
        logger.info(f"Configured task '{task_name}' with splits: {eval_splits}")
        
        # V2: Evaluate with proper v2 parameters
        results = mteb.evaluate(
            model,                                    # Custom model implementing EncoderProtocol
            tasks=[task],                             # Task object with configured splits
            cache=cache,                              # ResultCache for result management
            encode_kwargs={"batch_size": 16384},      # Encoding parameters
        )
        
        logger.info(f"Completed task: {task_name}")
        
        # ========================================================================
        # PRESERVED: Commented-out functionality from v1
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
        
        breakpoint()

    end = time.time()
    print(f"Time taken (mins): {(end-start)/60}")
    
    logger.info(f"\nResults saved to cache at: {cache_path}")