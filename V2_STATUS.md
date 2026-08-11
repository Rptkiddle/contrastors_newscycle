# NewsCycle V2 — Status and Onboarding

**Last updated**: 2026-08-11

## What is NewsCycle?

NewsCycle is a supervised contrastive fine-tune of `nomic-ai/nomic-embed-text-v1-unsupervised` (137M params) for **temporal-entity news retrieval** — the task of retrieving news articles about a specific entity from a specific time period. It is trained using Nomic's open-source `contrastors` pipeline, integrated alongside Nomic's existing fine-tuning datasets (MSMARCO, NLI, Reddit, etc.) as one additional dataset in the mixture.

The novel technical contribution is **temporally-aware hard-negative selection**: negatives are constrained to different month-year timepoints of the same entity, forcing the model to learn temporal discrimination rather than just entity discrimination.

## How this fork differs from upstream Nomic contrastors

**New files (the NewsCycle additions):**
- `scripts/text/get_negatives_news.py` — temporal-constraint hard-negative mining (rejects candidates from the same month-year as the positive; uses iterative search expansion instead of random fill)
- `scripts/text/generate_counts_offsets.py` — creates `counts.json` + `offsets.json.gz` metadata required by the training data loader (upstream equivalent is S3-only)
- `src/contrastors/configs/data/finetune_triplets_news_{inter,extra,merged}.yaml` — data configs that add a NewsCycle shard alongside Nomic's 10 upstream datasets
- `src/contrastors/configs/train/contrastive_finetune_news.yaml` — training config with `document_max_length: 2048` and `query_max_length: 64` for NewsCycle's longer documents
- `convert_hf_to_st.py` — HF NomicBertModel → SentenceTransformer conversion

**Modified files (minimal, necessary changes):**
- `src/contrastors/configs/train/contrastive_finetune.yaml` — output dir, workers, data config path for Snellius
- `src/contrastors/layers/attention.py`, `models/encoder/modeling_nomic_bert.py`, `models/biencoder/modeling_biencoder.py`, `layers/moe.py`, `models/biencoder/flash_llama.py`, `flash_pythia.py` — `unpad_input` signature fix for newer flash-attn (numerically equivalent)
- `src/contrastors/dataset/text_text_loader.py` — S3 + local filesystem routing so the loader can read local NewsCycle shards alongside Nomic's S3-hosted datasets
- `src/contrastors/models/biencoder/modeling_biencoder.py` — `_tied_weights_keys` for transformers 5.x compatibility
- `src/contrastors/trainers/__init__.py` — guarded optional imports for partial installs
- `src/contrastors/eval/mteb_eval/*.py` — argparse refactoring, filepath handling
- `convert_hf_to_st.py` — `trust_remote_code` propagation fix
- `.gitignore`, `requirements.txt`

**Unchanged**: all core training logic (loss functions, GradCache, data loading, model architecture). We add a dataset to the mixture; we do not modify the training machinery.

## Repository layout

Three repos under `code/2_NEWS/wp32-embeds/`, each with `main` (active V2) and `v1_dissertation` (archived V1) branches:

| Repo | Purpose | main branch | v1_dissertation branch |
|---|---|---|---|
| `newscycle-contrastors/` | Nomic contrastors fork (this repo) | Active V2 development | V1 frozen fork (read-only) |
| `newscycle-gdelter/` | GDELT collection (`collect.py`) + processing (`process.py`) | V2 BigQuery collector + processor | V1 file-based downloader |
| `newscycle-paper/` | SLURMs, supplementary notebooks | V2 SLURM scripts + supplementary | V1 SLURM scripts + supplementary |

All three repos default to `main` (V2). To view V1: `git checkout v1_dissertation`.

## Data layout

All under `data/2_NEWS/wp32-embeds/`:

```
gdelt/                    # shared, version-agnostic
├── processed/            # 11G, 2,147 merged GDELT day-files (2020–2025). PROTECTED.
├── quality/              # domain_pc1.csv, domain_ratings.csv
├── logs/                 # pipeline run logs
└── gdelter.zip           # 9.3G safety backup (pending inspection)

v1/                       # V1 artifacts (corresponds to main branches)
├── training-data/        # 5 JSONL splits (train/test × inter/extra + merged)
├── processing/           # notebook exports from q:d construction
├── hard-negatives/       # k=7 HN shards (5.8G)
├── models/               # finetuned checkpoints (8.2G, canonical for dissertation)
└── eval/                 # MTEBv2, NewsCycle, DailyOracle, TempReason results

v2/                       # V2 artifacts (corresponds to v2 branches)
├── training-data/        # empty — will hold new q:d formulation output
├── hard-negatives/       # k=20 HN shards from first attempt (15G, reusable)
└── eval/                 # regression evidence from abandoned run

supplementary/            # ST-datasets for audit notebooks (CC_news, NPR, CNN, AG_news)
```

## Data pipeline

Training data comes from GDELT (Global Database of Events, Language, and Tone), 2020–2025:
- ~4.7M articles after domain quality filtering (Lin 2023, PC1 ≥ 0.75)
- Entity filtering: temporal frequency ≥ 12 months, mean document frequency ≥ 15 articles/month → 2,449 entities, 154,704 (entity, month, year) keys
- Collection: `newscycle-gdelter/collect.py` (BigQuery → per-day parquet)
- Processing: `newscycle-gdelter/process.py` (parquet → training JSONL)
- Notebook: `newscycle-gdelter/processing/gdelt_processing.ipynb` (interactive exploration)

**V1 q:d construction:** For each (entity, month) key, all matching articles are concatenated chronologically (title + description + entity-filtered verbphrases/quotes), date-masked, and capped at 2048 estimated tokens. Queries are natural-language templates combining entity name and month-year (960 variations).

**V2 q:d redesign (pending decision):** A separability probe (run in a prior session; results documented in project memory) tested atomic (single-article) vs aggregate (concatenated) document representations. Key findings:
- Aggregates retain a robust temporal signal (Cohen's d = 0.90 for within-vs-adjacent month)
- But in oracle retrieval — which mirrors training and eval — atomic candidates win (P@1 = 0.588 vs 0.527)
- Recommendation: adopt atomic documents for V2 (better distribution match to deployment, crisper hard negatives, no doc_max machinery), but note that aggregation is not broken — the switch is for practical advantages, not because the old design was geometrically defective

**Decision not yet made by the user.** This is the first thing to resolve before V2 training.

## V1 models (canonical, on HuggingFace)

| Model | HF ID | Split | Use |
|---|---|---|---|
| Interpolation | `rptkiddle/NewsCycle_inter_st` | 60/40 by month-year within 2020–2023 | Historical retrieval within known window |
| Extrapolation | `rptkiddle/NewsCycle_extra_st` | 2020–2023 train / 2024–2025 test | Generalisation to unseen future |
| Merged | `rptkiddle/NewsCycle_st` | All 154,704 keys | Production model (all data) |

V1 training used k=7 fixed hard negatives, doc_max_length=256 (dataloader default, not an intentional choice), no grad_cache.

## V1 → V2 regression: what happened

Three changes were bundled in the failed V2 attempt and caused a 30–60% regression on in-domain NewsCycle:

1. **k=7 → k=20 HN** (most suspect) — random sampling from a larger pool dilutes the gradient signal for temporal discrimination
2. **doc_max_length 256 → 2048** — the separability probe suggests this is NOT the primary cause (aggregates retain signal)
3. **grad_cache=true** — enabled to fit (2) in memory; disables autocast at inner step, possible numerical drift

Detailed regression numbers (V2 vs V1, same eval pipeline):

| Benchmark | Metric | V2 | V1 | Relative |
|---|---|---|---|---|
| NewsCycle inter | Recall@1 | 0.040 | 0.102 | −61% |
| NewsCycle inter | MRR | 0.100 | 0.231 | −57% |
| NewsCycle extra | Recall@1 | 0.047 | 0.070 | −34% |
| NewsCycle extra | MRR | 0.114 | 0.192 | −41% |
| DailyOracle | Acc@1 | 0.265 | 0.284 | −7% |

The V2 models have been deleted. The V2 k=20 hard-negative shards are preserved for isolation testing.

## Nomic ground-truth parameters (contrastive fine-tuning stage)

These are the parameters we reproduce. Source: Nussbaum et al. 2025, arxiv.org/abs/2402.01613.

- **Checkpoint**: `nomic-ai/nomic-embed-text-v1-unsupervised`
- **Batch size**: 256
- **Sequence length**: 2048 (but dataloader truncates to `DEFAULT_COL_TO_MAX_TOKENS = {query: 32, document: 256, negative: 256}`)
- **Learning rate**: 2e-5, AdamW (β₁=0.9, β₂=0.999), weight decay 0.01, grad clip 1.0
- **Warmup**: 400 steps, linear cooldown to 0
- **Epochs**: 1
- **Hard negatives**: k=7 from top-20 mined (gte-base embeddings, cosine similarity), randomly sampled at training time
- **Datasets**: MSMarco (484,864), NLI (275,200), Reddit (199,680), MEDI SuperNLI (177,408), HotpotQA (169,728), FEVER (139,776), MEDI StackExchange (100,352), NQ (69,888), MEDI Flickr (50,944), MEDI Wiki (24,832)
- **Checkpointing**: save every 4,500 steps
- **MTEB evaluation**: max seq 512, L2 normalisation for all tasks except Classification

## Model packaging recipe

Packaging converts BiEncoder checkpoint → HF NomicBertModel → SentenceTransformer format.

**Key details:**
- Uses **transformers 4.45.2** in an isolated venv (transformers 5.x has a `NomicBertModel.__init__` isinstance check that breaks the save/load round-trip)
- `convert_to_hf.py` then `convert_hf_to_st.py` with `--trust-remote-code --pooling mean --normalize`
- `push_to_hub(use_temp_dir=False)` requires a pre-created work directory matching the repo basename
- Post-conversion config corrections: `embd_pdrop=0.0`, `resid_pdrop=0.0`, `rotary_scaling_factor=2`
- Validation against `nomic-ai/nomic-embed-text-v1` reference (file manifest, config structure, modules layout)

## MTEB evaluation recipe

```python
NOMIC_PROMPTS = {
    "STS":                "classification: ",
    "Classification":     "classification: ",
    "PairClassification": "classification: ",
    "Reranking":          "classification: ",
    "Summarization":      "classification: ",
    "Clustering":         "clustering: ",
    "Retrieval-query":    "search_query: ",
    "Retrieval-document": "search_document: ",
}

model = SentenceTransformer(MODEL_ID, prompts=NOMIC_PROMPTS, trust_remote_code=True)
result = mteb.evaluate(model, [task], encode_kwargs={...}, cache=None)
```

**Critical**: prompts at construction time, task-type keys (not prompt-type keys), `cache=None` for fresh evaluation.

## Snellius HPC notes

- **Partition**: `gpu_h100` (4×H100, ~1 hr per training run)
- **Toolchain**: Python 3.13, CUDA 12.8, PyTorch 2.7.0, flash-attn 2.8.3
- **HN mining toolchain** (separate, older): Python 3.11.3, CUDA 12.1.1, PyTorch 2.1.2, faiss-cpu 1.7.4
- **SSH host key** may need re-accepting on current machine
- **Log hygiene**: successful SLURM logs go to `~/data/<stage>/logs/`; failed logs are deleted. Never leave `.out` files in `~/`.
- **Dataset cache isolation**: `HF_DATASETS_CACHE="$TMPDIR/hf_datasets"` per job to prevent parallel race conditions on ArguAna load

## Manuscript target

**TACL** (Transactions of the ACL). NLP methods paper framing. Four contributions:
1. NewsCycle training task (temporally-aware hard-negative selection)
2. NewsCycle benchmark (interpolation + extrapolation splits)
3. "Preserves vs erodes" finding — NewsCycle fine-tuning preserves temporal reasoning that standard retrieval fine-tuning (Nomic's recipe) erodes
4. Fully-open reproducible artifact chain

Manuscript at `manu/2_NEWS/wp32-embeds/v1/`, synced with Overleaf. See `project_wp32_framing.md` in the Claude memory directory for full framing details.

## What to do next

1. **Decide atomic vs aggregate q:d design** — probe results are ready, user hasn't made the call yet
2. **Implement chosen q:d design** in `newscycle-gdelter/processing/gdelt_processing.ipynb`
3. **Isolation testing** of k, doc_max, grad_cache on Snellius (one change per run)
4. **V2 model training** with final recipe
5. **TACL manuscript update** — template conversion, bootstrap CIs, results refresh

## GitHub repos

| Repo | GitHub | Visibility |
|---|---|---|
| `newscycle-contrastors` | `Rptkiddle/newscycle-contrastors` | Public (fork of nomic-ai/contrastors) |
| `newscycle-gdelter` | `Rptkiddle/newscycle-gdelter` | Private |
| `newscycle-paper` | `Rptkiddle/newscycle-paper` | Private |
| `NewsCycle` | `Rptkiddle/NewsCycle` | Empty placeholder, Apache 2.0 |
