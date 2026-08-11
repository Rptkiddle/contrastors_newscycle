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
- `REBUILD_NOTES.md` — comprehensive historical record of the V1 rebuild process
- `V2_STATUS.md` — this file

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

## Data pipeline

Training data comes from GDELT (Global Database of Events, Language, and Tone), 2020–2025:
- ~4.7M articles after domain quality filtering (Lin 2023, PC1 ≥ 0.75)
- Entity filtering: temporal frequency ≥ 12 months, mean document frequency ≥ 15 articles/month → 2,449 entities, 154,704 (entity, month, year) keys
- Data processing code: `../wp32-gdelt-downloader/02_gdelt_processing/gdelt_processing.ipynb`

**Current q:d construction (V1):** For each (entity, month) key, all matching articles are concatenated chronologically (title + description + entity-filtered verbphrases/quotes), date-masked, and capped at 2048 estimated tokens. Queries are natural-language templates combining entity name and month-year (960 variations).

**V2 q:d redesign (pending decision):** A separability probe (results in `../diagnostic/output/`) tested atomic (single-article) vs aggregate (concatenated) document representations. Key findings:
- Aggregates retain a robust temporal signal (Cohen's d = 0.90 for within-vs-adjacent month)
- But in oracle retrieval — which mirrors training and eval — atomic candidates win (P@1 = 0.588 vs 0.527)
- Recommendation: adopt atomic documents for V2 (better distribution match to deployment, crisper hard negatives, no doc_max machinery), but note that aggregation is not broken — the switch is for practical advantages, not because the old design was geometrically defective

**Decision not yet made by the user.** This is the first thing to resolve before V2 training.

## Branch layout

- **`main`** — frozen V1 fork. Source of the published models on HF. Read-only; never push to it.
- **`v2`** (this branch) — started from Nomic upstream. Contains code-quality improvements + the V1 rebuild pipeline. The V2 *training procedure* changes (k=20 HN, doc_max=2048, grad_cache) were tested and regressed the model 30–60% on in-domain NewsCycle (see REBUILD_NOTES.md "AIM1 outcome — final"). Those V2 models are abandoned. The code is preserved and is the starting point for the next V2 attempt.

## V1 models (canonical, on HuggingFace)

| Model | HF ID | Split | Use |
|---|---|---|---|
| Interpolation | `rptkiddle/NewsCycle_inter_st` | 60/40 by month-year within 2020–2023 | Historical retrieval within known window |
| Extrapolation | `rptkiddle/NewsCycle_extra_st` | 2020–2023 train / 2024–2025 test | Generalisation to unseen future |
| Merged | `rptkiddle/NewsCycle_st` | All 154,704 keys | Production model (all data) |

These are the manuscript-canonical models. V1 training used k=7 fixed hard negatives, doc_max_length=256 (dataloader default, not an intentional choice), no grad_cache.

## V1 → V2 regression: what we know

Three changes were bundled in the failed V2 attempt:
1. **k=7 → k=20 HN** (most suspect) — random sampling from a larger pool dilutes the gradient signal for temporal discrimination
2. **doc_max_length 256 → 2048** — the separability probe suggests this is NOT the primary cause (aggregates retain signal)
3. **grad_cache=true** — enabled to fit (2) in memory; disables autocast at inner step, possible numerical drift

**V2 plan**: test these changes in isolation (one per run, ~1 hr each on 4×H100) to identify which caused the regression, then build the V2 recipe from what works. This isolation testing should happen on top of whichever q:d design (atomic or aggregate) is chosen.

## MTEB evaluation recipe

The correct prefix recipe for any Nomic-family model evaluation:

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

Prompts must be passed at construction time. Keys are MTEB task types, not prompt types. `cache=None` is required for fresh evaluation (the default cache silently returns stale results).

## Key paths (all under `~/Desktop/`)

| Resource | Path |
|---|---|
| This repo | `code/2_NEWS/wp32-embeds/wp32-contrastors-newscycle/` |
| GDELT downloader + processing | `code/2_NEWS/wp32-embeds/wp32-gdelt-downloader/` |
| SLURM scripts | `code/2_NEWS/wp32-embeds/wp32-SLURMs/` |
| Separability probe | `code/2_NEWS/wp32-embeds/diagnostic/` |
| TACL manuscript (Overleaf) | `manu/2_NEWS/wp32-embeds/v1/` |
| Pipeline data | `data/2_NEWS/wp32-embeds/` |
| Project home (session home) | `proj/2_NEWS/wp32-embeds/` |

## Manuscript target

**TACL** (Transactions of the ACL). NLP methods paper framing. Four contributions:
1. NewsCycle training task (temporally-aware hard-negative selection)
2. NewsCycle benchmark (interpolation + extrapolation splits)
3. "Preserves vs erodes" finding — NewsCycle fine-tuning preserves temporal reasoning that standard retrieval fine-tuning (Nomic's recipe) erodes
4. Fully-open reproducible artifact chain

See `project_wp32_framing.md` in the Claude memory directory for full framing details.

## What to do next

1. **Decide atomic vs aggregate q:d design** — probe results are ready, user hasn't made the call yet
2. **Implement chosen q:d design** in `gdelt_processing.ipynb`
3. **Isolation testing** of k, doc_max, grad_cache on Snellius (one change per run)
4. **V2 model training** with final recipe
5. **TACL manuscript update** — template conversion, bootstrap CIs, results refresh

## Historical record

`REBUILD_NOTES.md` in this branch contains the full chronological record of the V1 rebuild process, including all findings, decisions, and benchmarks. It is comprehensive but long (~620 lines) and contains some stale paths (`1_NEWS` → `2_NEWS`). Consult it for historical context; use this file for current state.
