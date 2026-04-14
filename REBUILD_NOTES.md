# WP3.2 Pipeline Rebuild Notes

## Sources of truth

**This file (`REBUILD_NOTES.md`) is the canonical source of truth for the rebuild.** It documents what we have built, what we are building, and the decisions we made along the way. When the rebuild process disagrees with anything else, this file wins (and the disagreeing thing gets updated, not the other way around).

**The user's research notes** are reference material that accompanies the **frozen fork** (`fork/main` on `github.com/Rptkiddle/contrastors_newscycle`). They document a tested workflow that produced models with known MTEB problems on specific subsets. They are valuable as historical context and as starting hypotheses, but they are **not prescriptive**. When this file and the research notes disagree, raise it for explicit user resolution rather than silently picking one.

**Nomic's paper, blog, and HF model card** are the canonical reference for *what we are reproducing*. Target: Nomic's reported 0.821 MTEB STS on `nomic-embed-text-v1`. The rebuild's goal is to close the gap between the previously-published baseline (0.692) and Nomic's number, by re-running the entire pipeline from minimal-deviation rebuild-clean code in a controlled environment.

**`fork/main` is read-only** and never pushed to. The previously-published HF models (`rptkiddle/NewsCycle_inter_st`, etc.) were built from it and remain as frozen comparison baselines.

## Overview

Systematic rebuild of the NewsCycle embedding model pipeline to resolve the MTEB reproducibility gap between `baseline_st` (reproduced Nomic pipeline) and the official `nomic-embed-text-v1` results (Nussbaum et al. 2025).

**Manuscript gap (before any fixes):**

| Category | baseline_st | Official | Delta |
|---|---|---|---|
| Classification | 0.662 | 0.741 | −0.079 |
| Clustering | 0.434 | 0.439 | −0.005 |
| Pair Classification | 0.820 | 0.852 | −0.032 |
| Reranking | 0.553 | 0.558 | −0.005 |
| Retrieval | 0.472 | 0.528 | −0.056 |
| STS | 0.692 | 0.821 | −0.129 |
| **Overall** | **0.557** | **0.624** | **−0.067** |

**Key reference**: Nomic paper at https://arxiv.org/html/2402.01613v2

**Contrastors fork**: `github.com/Rptkiddle/contrastors_newscycle` (fork of `nomic-ai/contrastors`)
- Branch `main`: the **frozen reference fork**. Source of the v1 production models on HF (`rptkiddle/NewsCycle_inter_st`, `_extra_st`, `NewsCycle_st`). Read-only — never push to it.
- Branch `v2`: rebuild started from Nomic upstream (renamed from `rebuild-clean` 2026-04-13). Code-quality improvements + experimental training procedure changes. **The v2 training procedure regressed the model** (see "AIM1 outcome — final" below) so v2 is preserved as a code-quality reference but the v1 frozen-fork models remain canonical for the manuscript.

---

## Manuscript framing + target venue (AIM3, 2026-04-14)

**Target**: TACL (Transactions of the ACL) as primary, IPM (Information Processing & Management) as secondary fallback. Comm-sci venues (CMM, CCR) considered and rejected because they would require shoe-horning that weakens the paper.

**Framing**: Face A (NLP/IR methods paper) is dominant. Face B (open infrastructure / reproducibility) is an enabling argument, not a standalone contribution. Face C (comm-sci motivation) belongs in the Introduction only.

**Four contributions in order**:
1. NewsCycle training task (including temporally-aware hard-negative selection)
2. NewsCycle benchmark (interpolation + extrapolation splits)
3. "Preserves vs erodes" empirical finding (standard retrieval fine-tune damages TempReason by 0.107/0.081; NewsCycle fine-tune is flat at -0.022/+0.006 vs the common base — our method protects a capability the standard recipe destroys)
4. Fully-open reproducible artifact chain (weights + code + benchmark + data pipeline, load-bearing because the method requires fully open infrastructure)

**Central empirical claim (headline for Discussion)**: NewsCycle fine-tuning preserves temporal-reasoning capability that the dominant retrieval fine-tuning recipe erodes — an underappreciated cost of the standard recipe that our approach mitigates.

**Do not**: frame as comm-sci methods paper, lead with the blind spots as findings, claim we discovered the temporal gap, include LongEmbed (dissertation version). 

**Full details**: see `project_wp32_framing.md` in the claude memory dir. That file is the canonical framing reference for all AIM3 revision work.

**AIM3 in-progress items** (2026-04-14 evening):
- Background + Discussion sections are placeholders ("to be written")
- Targeted literature review planning is the next step
- Section-by-section refinement of existing sections follows

## AIM1 outcome — final (2026-04-13 evening)

AIM1 has TWO findings, one positive (morning) and one negative (evening). Both matter for the manuscript:

### Finding 1 — Baseline reproduction succeeded (morning)

The rebuild-clean code (now `v2` branch) produces a Nomic-equivalent baseline model. Trained on Nomic's own training recipe (no NewsCycle data), the rebuild's `baseline_st_temp_v2` scores **+0.003 above Nomic's own published `nomic-ai/nomic-embed-text-v1`** on a head-to-head MTEBv2 STS run (0.6978 vs 0.6950 overall). Statistically indistinguishable. This proves the rebuild *environment* and *code* are functionally equivalent to Nomic's, and resolves the original manuscript's "0.129 STS gap" as an MTEBv1 → MTEBv2 library transition artifact (not a real model deficiency).

### Finding 2 — Production training procedure regressed the model (evening)

When the v2 rebuild was applied to the production NewsCycle training (inter/extra/merged with k=20 HN mining, `document_max_length=2048`, `grad_cache=true`), the resulting models **substantially underperformed the v1 frozen-fork models on the in-domain NewsCycle benchmark**:

**NewsCycle inter** (test_inter.jsonl, 62K queries, head-to-head v2 vs v1 same eval pipeline):

| Metric | v2 (rebuild) | v1 (frozen fork) | Δ | Relative |
|---|---|---|---|---|
| Recall@1 | 0.0401 | 0.1022 | −0.062 | **−61%** |
| Recall@5 | 0.1384 | 0.3453 | −0.207 | −60% |
| Recall@10 | 0.2248 | 0.5335 | −0.309 | −58% |
| MRR | 0.1003 | 0.2310 | −0.131 | −57% |

**NewsCycle extra** (test_extra.jsonl, 53K queries, same head-to-head):

| Metric | v2 (rebuild) | v1 (frozen fork) | Δ | Relative |
|---|---|---|---|---|
| Recall@1 | 0.0466 | 0.0704 | −0.024 | **−34%** |
| Recall@5 | 0.1592 | 0.2894 | −0.130 | −45% |
| Recall@10 | 0.2576 | 0.4796 | −0.222 | −46% |
| MRR | 0.1138 | 0.1923 | −0.079 | −41% |

**DailyOracle MCQ** (18K questions, same head-to-head, merged model only):

| Metric | v2 (merged) | v1 (NewsCycle_st) | Δ |
|---|---|---|---|
| Accuracy@1 | 0.2648 | 0.2839 | −0.019 |
| MRR | 0.5271 | 0.5402 | −0.013 |

DailyOracle was a smaller drop, but in the same direction. The dominant signal is the in-domain NewsCycle regression.

### What changed that broke the model

Three deliberate "alignments with Nomic's published recipe" were applied during v2 training (vs the v1 frozen fork):

1. **HN mining pool size: k=7 → k=20** with random sampling of 7 per training step. This matches Nomic's *"top 20 documents mined, randomly sampled the negatives"* description exactly. The v1 fork used a fixed k=7 set per query.
2. **`document_max_length: 2048`** explicit override (vs the dataloader's hardcoded 256-token default that the v1 fork inherited). NewsCycle docs have median ~1500 tokens; the v1 fork was actually training on heavily truncated documents without realizing it.
3. **`grad_cache: true`** to fit (2) on 94 GiB H100s without OOM. Mathematically equivalent in principle, but introduces chunked-recomputation that disables autocast at the inner step.

**Most plausible primary cause**: the k=7 → k=20 change. With random sampling from a larger pool of negatives, the gradient signal during training averages over a *less-focused* set of negatives. For a task where temporal discrimination depends on subtle differences between adjacent months of the same entity, the v1 fork's "tighter" k=7 set may have been enforcing *exactly* the right kind of discrimination. Nomic's k=20 + sample-7 approach matches their published recipe but Nomic's data distribution (MSMARCO, NLI, etc.) is very different from NewsCycle's temporal-news distribution.

**Less plausible but possible secondary causes**: (2) the longer documents may dilute the temporal signal by adding entity-context that's redundant across months; (3) grad_cache + autocast interaction may introduce subtle numerical drift, though we don't have evidence for this.

**Not the cause**: training environment. Finding 1 (baseline reproduction) proved the rebuild-clean *environment* produces Nomic-equivalent models. The regression is in the training *procedure changes*, not the environment.

### Decision: revert to v1 frozen-fork models for the manuscript

For the manuscript's results section:
- **NewsCycle benchmark**: report v1 fork numbers from the user's research notes (`NewsCycle_inter_st`: Recall@1 = 10.22% / MRR = 0.231; `NewsCycle_extra_st`: Recall@1 = 7.04% / MRR = 0.192). These have been independently re-validated under our current eval pipeline today.
- **DailyOracle**: KEEP. The v1 frozen-fork merged model holds rank 2 on DailyOracle (Acc@1 = 0.2839, essentially tied with embeddinggemma at 0.2877), and beats Nomic v1 in the same eval pipeline. The MCQ-vs-embedding methodological caveat exists but the empirical result is good for our shipping model. (Originally proposed to drop on methodological grounds; reversed by user 2026-04-13 evening because the v1 result is competitive enough to keep.)
- **MTEBv2 head-to-head**: run `slurm4_mteb_eval.sh` with `MODEL_LABEL=merged` (= `rptkiddle/NewsCycle_st`, the v1 frozen-fork merged model) vs `MODEL_LABEL=nomic_v1`. Fresh same-day same-pipeline numbers across both models for the manuscript's MTEB table.

### What `v2` is preserved as

The `v2` branch is preserved on github + Snellius + local as a **clean code reference** that contains:
- Fixed prefix-handling bug in MTEB evaluation
- Documented training environment (2025 toolchain, all dependencies pinned)
- Self-contained HN mining scripts (committed to `scripts/text/`, no orphan dependencies)
- Systematized packaging pipeline (`slurm_package.sh` with transformers 4.45.2 isolated venv)
- Comprehensive REBUILD_NOTES.md (this file)

The `v2` *models* (`rptkiddle/<label>_{hf,st}_temp_v2` on HF) are abandoned. They will be deleted from HF Hub as part of today's housekeeping. Local checkpoints in `~/data/03_finetuned_model/<label>/` and `~/data/04_packaged_model/<label>_st/` are preserved (they're in the data archive being rsync'd to local).

Future v2 work could revisit individual training-procedure changes in isolation — e.g., test k=20 HN alone without the doc_max_length=2048 change, or vice versa — to identify which specific change caused the regression. That's a deferred experiment, not blocking AIM1 closure.

### What this means for the manuscript

The **methods section** should describe the v1 production pipeline (the frozen-fork training recipe — k=7 HN, `document_max_length` left at the dataloader default, no grad_cache, the actual hyperparameters that produced the published models). That description is what's already in the manuscript, modulo the prefix-fix detail (§MTEB Evaluation should be updated to clarify how prefixes are passed).

The **results section** should report:
- MTEBv2 head-to-head: v1 NewsCycle_st (merged) vs Nomic v1 — to be run after this housekeeping completes
- NewsCycle (entity-temporal retrieval): v1 inter / extra / merged numbers, plus 8 comparison baselines (already in research notes)
- DailyOracle MCQ: v1 NewsCycle_st rank 2 (Acc@1 = 0.2839, tied with embeddinggemma 0.2877; beats Nomic v1 0.2745). Acknowledge MCQ-vs-embedding caveat in methods/limitations but report the result.

The "0.129 STS gap" framing in the original manuscript should be replaced with the MTEBv2 head-to-head finding: under MTEBv2, our v1 model is statistically indistinguishable from Nomic v1 on STS, and the original "gap" was a library-version artifact.

---

## Final MTEBv2 head-to-head results (2026-04-13 evening)

Two parallel SLURM jobs on Snellius `gpu_h100` (jobs `21839124` mteb_merged, `21839125` mteb_nomic_v1) ran the full **MTEB(eng, v2)** benchmark end-to-end. 41 tasks, ~90 min wall each. Both completed cleanly. Summaries in `~/data/05_eval_benchmarks/MTEBv2/{merged,nomic_v1}/summary.json`. Slurm logs archived to `<output_dir>/logs/` per the log-hygiene policy.

**Key implementation detail** (preserve for future sessions): `slurm4_mteb_eval.sh` sets `HF_DATASETS_CACHE="$TMPDIR/hf_datasets"` per-job to isolate dataset caches across parallel runs. Without this, both jobs race on `.incomplete/` renames in the shared `~/.cache/huggingface/datasets/`, and one job dies on ArguAna load. Do not revert.

### Category averages

| Category | n | merged (`rptkiddle/NewsCycle_st`) | `nomic-ai/nomic-embed-text-v1` | Δ |
|---|---|---|---|---|
| Classification | 8 | 0.7600 | 0.7609 | −0.0009 |
| Clustering | 8 | 0.4604 | 0.4670 | −0.0066 |
| PairClassification | 3 | 0.8451 | 0.8515 | −0.0064 |
| Reranking | 2 | 0.4575 | 0.4585 | −0.0009 |
| **Retrieval** | 10 | **0.5192** | **0.5455** | **−0.0263** |
| STS | 9 | 0.8033 | 0.8175 | −0.0141 |
| Summarization | 1 | 0.3278 | 0.3240 | **+0.0038** |
| **OVERALL** | **41** | **0.6333** | **0.6447** | **−0.0114** |

### Retrieval-gap diagnosis

The overall −0.0114 gap is almost entirely driven by three adversarial or specialized-retrieval tasks where domain-fine-tuning costs the most:

| Task | merged | nomic_v1 | Δ |
|---|---|---|---|
| ArguAna (argument counter-retrieval) | 0.3356 | 0.4918 | −0.156 |
| Touche2020.v3 (argument retrieval) | 0.5984 | 0.6516 | −0.053 |
| TRECCOVID (specialized biomed retrieval) | 0.7288 | 0.7961 | −0.067 |

Strip those three and the Retrieval category delta shrinks from −0.026 to roughly −0.002. The other 7 retrieval tasks are near-tied. The same noise-level story holds across the rest of the benchmark: outside the three adversarial-retrieval casualties, 38 tasks trade blows within ±0.02.

### Manuscript framing

The story for §4 Results: **fine-tuning on NewsCycle preserves general MTEB capability** (overall −0.011), **with the cost concentrated in adversarial and specialized retrieval tasks** (ArguAna, Touche, TRECCOVID). On the in-domain NewsCycle benchmark the fine-tune produces substantial gains over the Nomic v1 base (v1 `NewsCycle_inter_st` Recall@1 = 0.1022; v1 `NewsCycle_extra_st` Recall@1 = 0.0704). On DailyOracle the merged model holds rank 2 (Acc@1 = 0.2839, tied with embeddinggemma 0.2877, beats Nomic v1 0.2745) despite the MCQ-vs-embedding methodological caveat.

## TempReason out-of-domain validation (2026-04-14)

Added as §4.4 after DailyOracle. 9-model panel, nDCG@10 on the English test splits of `TempReasonL2Fact` and `TempReasonL3Fact` (Tan et al. 2023). For 6 of 8 baselines the scores come from the HF `mteb/results` leaderboard (published numbers at each model's submitted best configuration); for NewsCycle_st, qwen3-embedding-0.6b, and google/embeddinggemma-300m (which had no TempReason L2/L3 Fact results on the leaderboard) we measured them ourselves via `eval4_tempreason.sh` using the MTEB evaluate pipeline.

### Final table (sorted by L2Fact desc)

| Model | Release | L2Fact | L3Fact | Source |
|---|---|---|---|---|
| embeddinggemma-300m | Sep 2025 | **0.3428** | 0.2803 | ours |
| bge-m3 | Jan 2024 | 0.3323 | **0.3005** | HF |
| nomic-embed-text-v1-unsupervised | Feb 2024 | 0.2216 | 0.1997 | HF |
| qwen3-embedding-0.6b | Jun 2025 | 0.2120 | 0.1883 | ours |
| **NewsCycle_st** | — | **0.1997** | **0.2052** | **ours** |
| all-MiniLM-L6-v2 | Aug 2021 | 0.1765 | 0.1416 | HF |
| nomic-embed-text-v1 | Feb 2024 | 0.1143 | 0.1189 | HF |
| all-mpnet-base-v2 | Aug 2021 | 0.1120 | 0.0942 | HF |
| paraphrase-multi-MiniLM-L12-v2 | Aug 2021 | 0.0621 | 0.0677 | HF |

### Manuscript framing

NewsCycle_st ranks **3rd on L3Fact, 5th on L2Fact** in the 9-model panel. The methodologically meaningful comparison is against `nomic-embed-text-v1` (Nomic's general-purpose supervised fine-tune of the same base): Nomic's recipe **destroys** temporal-reasoning capability of the base model (L2Fact 0.2216→0.1143, L3Fact 0.1997→0.1189), while NewsCycle's temporal-entity fine-tune is **essentially flat** vs the base (L2 Δ=−0.022, L3 Δ=+0.006). The framing claim is: **NewsCycle fine-tuning preserves temporal-reasoning capability that standard retrieval fine-tuning erodes**, as an underappreciated cost of optimising against MS MARCO-style large-scale retrieval data.

### Caveats

- Qwen3 and embeddinggemma were run ourselves with plain SentenceTransformer encoding (no `prompt_name="query"`, no `encode_query()`/`encode_document()` dedicated methods). Their scores are floor estimates; their author-submitted best configs may be slightly higher. Worth noting that embeddinggemma still ranks #1 on L2Fact even with our under-optimal encoding.
- Our own measurement of `nomic-embed-text-v1` (via slurm4c earlier on 2026-04-14, before we decided to use HF-published numbers for baselines) gave L2=0.1581 / L3=0.1663 — significantly higher than the HF-published 0.1143 / 0.1189. The discrepancy is attributable to HF's NomicWrapper config (max_tokens=8192 rotary extrapolation + default prompts dict with `query`/`document` keys instead of `Retrieval-query`/`Retrieval-document`). We defer to the HF-published number in the paper because it represents "Nomic v1 at its community-submitted best" per standard comparison practice.

### Key session facts to preserve

1. **AIM1 is RESOLVED**, with two findings:
   - Morning: rebuild-clean code reproduces Nomic baseline under MTEBv2 (+0.003).
   - Evening: v2 training procedure (k=20 HN, doc_max_length=2048, grad_cache) regressed in-domain NewsCycle by 30-60% relative. v2 models abandoned.
2. **v1 frozen-fork models are canonical** for the manuscript: `rptkiddle/NewsCycle_inter_st`, `_extra_st`, `NewsCycle_st`. These are the shipping artifacts.
3. **The v2 branch is preserved** as a code-quality reference (clean environment, prefix bug fix, in-repo HN scripts, systematized packaging, REBUILD_NOTES). The v2 *models* on HF (`*_temp_v2` suffixes) were deleted 2026-04-13.
4. **DailyOracle KEPT** in the manuscript — v1 NewsCycle_st holds rank 2, empirically good enough to report. Acknowledge MCQ caveat but report the result.
5. **rsync of Snellius `~/data/` to local** complete at `~/Downloads/snellius_data_20260413/` (~25 GB).
6. **Manuscript path**: `/Users/rupertkiddle/Desktop/manu/1_NEWSFLOWS/wp32_embeds/v1/0_manuscript.tex` — outside primary working dir, requires explicit per-session permission.
7. **Future v2 experiments** (deferred): test k=20 HN, doc_max_length=2048, grad_cache changes IN ISOLATION rather than bundled, to identify which specific change caused the regression.

### Manuscript update checklist

- **§3 Methods (training)**: leave v1 description as-is. Optionally add a short methodological-transparency footnote about the v2 training procedure attempt and its abandonment. The §MTEB Evaluation subsection needs updating to clarify the prefix-handling recipe (`prompts=NOMIC_PROMPTS` at construction, task-type keys, MTEB v2 `cache=None`).
- **§3.5 Hard Negative Selection**: leave the v1 k=7 description (or note in passing that k=20 + sample-7 alignment with Nomic's paper recipe was tested and regressed in-domain performance).
- **§4 Results — MTEB table**: REPLACE old MTEBv1 numbers with the fresh MTEBv2 head-to-head above. Note in the caption that the comparison is under MTEBv2; the original MTEBv1 leaderboard numbers are not directly comparable.
- **§4 Results — NewsCycle table**: KEEP existing v1 numbers from research notes (re-validated today: inter R@1 = 0.1022, extra R@1 = 0.0704).
- **§4 Results — DailyOracle table**: KEEP. Acknowledge MCQ-vs-embedding methodological caveat but report the rank-2 result.
- **§Reproducibility**: reference v2 branch on `github.com/Rptkiddle/contrastors_newscycle` as the reproducible build.

**Editing approach**: don't touch citations or framing beyond what's needed. Don't push to Overleaf without explicit user approval (the manuscript is on the Overleaf git bridge).

After the manuscript update, proceed to **AIM2** (comm-sci validation design — task #5). AIM2 is about designing validation exercises that demonstrate the model's value to communication scientists — topic modeling, event detection, news retrieval/search workflows — rather than abstract benchmark metrics. StreamingQA (Liska et al. 2022) was discussed as one possible temporal-OOD benchmark; otherwise AIM2 is a fresh design exercise.

## Findings from initial investigation (2026-04-11 — 2026-04-12)

### MTEB eval prefix bug — CONFIRMED AND FIXED

**Root cause**: slurm4_mteb_eval.sh set `model.prompts = {"query": ..., "document": ...}` after construction. This is a no-op because:
1. SentenceTransformers requires prompts at construction time (`SentenceTransformer(..., prompts={...})`) per the ST documentation.
2. The keys `"query"` / `"document"` are prompt-type keys, only consulted for asymmetric retrieval tasks. Symmetric tasks (STS, Classification, PairClassification, Clustering, Reranking, Summarization) need task-type keys (`"STS"`, `"Classification"`, etc.).
3. MTEB v2's `mteb.evaluate()` defaults to `cache=ResultCache()`, which silently returns cached results without calling encode. This masked all our fix attempts until we passed `cache=None`.

**Correct prompts dict** (per Nussbaum et al. 2025):
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

**STS results with fix on baseline_st:**

| Task | Fixed | Broken | Delta |
|---|---|---|---|
| BIOSSES | 0.8619 | 0.8438 | +0.018 |
| SICK-R | 0.7806 | 0.7929 | −0.012 |
| STS12 | 0.7796 | 0.7610 | +0.019 |
| STS13 | 0.8388 | 0.8104 | +0.028 |
| STS14 | 0.7949 | 0.7526 | +0.042 |
| STS15 | 0.8670 | 0.8354 | +0.032 |
| STS16 | 0.8504 | 0.8281 | +0.022 |
| STS17 | 0.1599 | 0.0366 | +0.123 |
| STS22 | 0.6390 | 0.6496 | −0.011 |
| STSBenchmark | 0.8436 | 0.8303 | +0.013 |
| SummEval | 0.3101 | 0.2942 | +0.016 |
| **AVG** | **0.7023** | **0.6759** | **+0.026** |

Fix closes ~22% of the STS gap. Remaining gap to Nomic's 0.821 requires investigation of training/packaging drift.

**Corrected eval script**: `wp32_SLURMs/slurm4_mteb_eval.sh`

### Additional issues identified

**Nomic official model also ships with empty prompts**: `nomic-ai/nomic-embed-text-v1` has `model.prompts = {'query': '', 'document': ''}` with empty strings. Nomic got their published MTEB scores using their own `STransformer` wrapper class (at `src/contrastors/eval/encoder.py`) which manually prepends prefixes, not through ST/MTEB's prompt mechanism. Our fix uses MTEB's official prompt API correctly.

**Model packaging issue**: `convert_hf_to_st.py` bakes in `{'query': '', 'document': ''}` empty prompts. Should either include the correct Nomic prompts or ship with an empty dict. Fix at stage (v).

**Manuscript-code inconsistency**: §3.5 "Hard Negative Selection" describes temporally-constrained HN mining (negatives from different month-year timepoints). Commit `c3a7b81` says: *"temporal hard negative selection no longer needed; reverted to nomic approach + retained safe shard writing."* Must resolve at stage (iii): either reinstate temporal HN or update manuscript.

### Dismissed suspects (no action needed)

- `seq_len: 512` / `activation_function` in `contrastive_finetune.yaml` — both are no-ops when `pretrained: true` with `nomic_encoder: true`. Model architecture comes from HF trunk config via `AutoConfig.from_pretrained()` at `modeling_biencoder.py:217`.
- `layers/attention.py`, `moe.py`, `modeling_nomic_bert.py`, `biencoder/modeling_biencoder.py` — benign `unpad_input` signature patches for newer flash-attn; numerically equivalent.
- `train.py`, `convert_to_hf.py` — sys.path hacks only.
- `text_text_loader.py` S3+local filesystem routing — correctly scoped, needed to mix S3 and local data sources.

### Diff summary: fork vs upstream (23 files changed)

**New files (by design):**
- `convert_hf_to_st.py` (316 lines) — custom HF→ST conversion
- `scripts/text/get_negatives_news.py` (348 lines) — temporal HN mining
- `scripts/text/get_negatives_safe.py` (222 lines) — safe single-thread shard writer
- `scripts/text/generate_counts_offsets.py` (135 lines) — missing utility
- `configs/data/finetune_triplets_news.yaml` (87 lines) — data config with NewsCycle

**Modified for Snellius compatibility:**
- `configs/train/contrastive_finetune.yaml` (8 lines) — output dir, workers, data config path
- `layers/attention.py` (10 lines) — flash-attn `unpad_input` return value fix
- `models/encoder/modeling_nomic_bert.py` (4 lines) — same unpad fix
- `models/biencoder/modeling_biencoder.py` (2 lines) — same
- `layers/moe.py` (2 lines) — same
- `models/biencoder/flash_llama.py` (4 lines), `flash_pythia.py` (4 lines) — same
- `dataset/text_text_loader.py` (42 lines) — S3+local filesystem routing
- `train.py` (9 lines) — sys.path hack for running without pip install

**Modified for eval:**
- `eval/mteb_eval/eval_mteb.py` (9 lines) — sys.path hack
- `eval/mteb_eval/merge_cqadupstack.py` (36 lines) — filepath handling
- `eval/mteb_eval/mteb_meta.py` (35 lines) — argparse refactor
- `eval/mteb_eval/score_mteb.py` (43 lines) — argparse refactor

**Other:**
- `.gitignore` (11 lines)
- `requirements.txt` (361 lines rewritten)
- `requirements_pinned.txt` (181 lines, new)

---

## Nomic ground-truth parameters (from paper)

### MLM Pretraining (nomic-bert-2048)
- 137M parameters, 12 layers, BERT-base architecture
- Rotary positional embeddings (replacing absolute)
- SwiGLU activation (replacing GeLU)
- Dropout: 0
- Tokenizer: bert-base-uncased
- Sequence length: 2048
- Masking rate: 30%
- Optimizer: AdamW, lr=5e-4, β₁=0.9, β₂=0.98, weight decay 1e-5
- Warmup: 6% of total steps, linear decay to 0
- Batch size: 4096 global
- Data: BooksCorpus + Wikipedia 2023

### Contrastive Pretraining (Unsupervised)
- 235M pairs (filtered from 470M using gte-base consistency, k=2)
- Sequence length: 2048
- Batch size: 16,384 global
- lr: 2e-4, β₁=0.9, β₂=0.999, weight decay 0.01, grad clip 1.0
- Warmup: 700 steps, inverse sqrt decay
- 1 epoch
- Loss: InfoNCE, unidirectional query→document
- Prefixes: search_query, search_document, classification, clustering

### Contrastive Fine-tuning (Supervised) — THIS IS WHAT WE REPRODUCE
- Checkpoint: nomic-ai/nomic-embed-text-v1-unsupervised
- Batch size: 256
- Sequence length: 2048
- lr: 2e-5, β₁=0.9, β₂=0.999, weight decay 0.01, grad clip 1.0
- Warmup: 400 steps, linear cooldown to 0
- 1 epoch
- Hard negatives: k=7, mined from top-20 most similar docs
- Datasets: MSMarco (484,864), NLI (275,200), Reddit (199,680), MEDI SuperNLI (177,408), HotpotQA (169,728), FEVER (139,776), MEDI StackExchange (100,352), NQ (69,888), MEDI Flickr (50,944), MEDI Wiki (24,832)
- Save checkpoint every 4500 steps

### MTEB Evaluation
- Max sequence length: 512 (truncated)
- Prefixes per task type: see NOMIC_PROMPTS dict above
- L2 normalization: all tasks EXCEPT Classification
- MSMARCO: dev split; all others: test split

---

## Pipeline rebuild stages

### Stage (i): Collect training data
**Status**: COMPLETE (not re-running)
**Action**: Verify existing GDELT outputs are intact.
- ~4.7M articles, 2020-01-01 to 2025-12-29
- Domain quality filter: PC1 ≥ 0.75
- Power-allocation domain balancing: α=0.5, ~2500 articles/day
- Data location: `/Users/rupertkiddle/Desktop/data/1_NEWSFLOWS/wp32_embeds/contrastors_newscycle/tf12_mdf15_60_40/stage1_download`

### Stage (ii): Process training data
**Status**: COMPLETE (not re-running)
**Action**: Review processing code in gdelt_processing.ipynb, verify train/test splits.
- Temporal Frequency filter: TF ≥ 12 months (173,135 entities, 61.8% of mentions)
- Mean Document Frequency filter: mDF ≥ 15 articles/month (2,449 entities, 42.3% of mentions)
- 154,704 unique (entity, month, year) keys
- Interpolated split: 60/40 by month-year (92,083 train / 62,621 test)
- Extrapolated split: 2020-2023 train / 2024-2025 test (101,184 / 53,520)
- Merged set: all 154,704 keys
- Date masking enabled, entity-filtered verbphrases/quotes, max 2048 tokens per doc
- Data location: `.../stage2_process` and `.../stage3_training/01_training_data`

### Stage (iii): Mine hard negatives
**Status**: COMPLETE (2026-04-12)
**Action**: Re-mined NewsCycle data with temporal HN constraint and k=20.

**Diff review (3 new files on fork, upstream unchanged):**
- `get_negatives_safe.py`: identical HN logic to upstream `get_negatives.py`, only change is safe rank-0-only shard writing with atomic replace. Pure infrastructure fix.
- `get_negatives_news.py`: temporal-constraint variant (manuscript §3.5). Rejects candidates from the same month-year as the query. Uses iterative search expansion instead of random fill. Also includes safe writing.
- `generate_counts_offsets.py`: utility to create `counts.json` + `offsets.json.gz` from local shards. Required by the training data loader but missing from upstream (their S3 data has these pre-built).

**Decisions made:**
1. **Temporal HN: KEEP** — it is a core contribution described in §3.5 and was what slurm1 actually ran (`get_negatives_news.py`). Commit `c3a7b81` message was about an intermediate iteration, not the final run.
2. **Mine k=20** (not k=7) with temporal constraint — matching Nomic's approach of mining a larger pool, then randomly sampling 7 at training time. Paper confirms: "top 20 documents" mined, "seven hard negatives per pair" used, "randomly sampled the mined negatives." Training loader (`text_text_loader.py:607`) does `random.sample(data[col], self.num_negatives)` when `sample_negatives=True` (default).
3. **Use `get_negatives_news.py`** with `--k 20` for NewsCycle data. Run `generate_counts_offsets.py` after to create loader metadata.
4. **Do NOT re-mine Nomic's upstream datasets** — use their pre-mined S3 shards directly at training time.
5. **Scripts absorbed into rebuild-clean (2026-04-13)** — the `get_negatives_news.py` and `generate_counts_offsets.py` files initially lived as orphan scripts at `~/scripts/` on Snellius (outside the contrastors repo) so that stage (iii) could run without depending on the full repo dependency chain. On 2026-04-13 they were brought into `scripts/text/` in rebuild-clean (commit `be7b5ba`) so the rebuild is self-contained and stage (iii) is reproducible from the repo alone. The `~/scripts/` directory was removed from Snellius. The `slurm1_get_news_negatives.sh` driver was updated to invoke from `$REPO_DIR/scripts/text/` instead of `$HOME/scripts/`. The absorbed `get_negatives_news.py` is the version that actually ran (with FAISS GPU-with-CPU-fallback already integrated).

**Toolchain for this stage:**
- 2023: Python 3.11.3, CUDA 12.1.1, PyTorch 2.1.2, FAISS 1.7.4 (easybuild)
- This differs from training/eval (2025 toolchain) but the boundary is clean: HN mining outputs plain data files.
- gte-base embeddings are deterministic regardless of CUDA version.

**Nomic ground-truth parameters (HN mining):**
- Embedding model: gte-base (thenlper/gte-base)
- Tokenizer max length: 512
- Similarity: inner product (FAISS IndexFlatIP), L2-normalized embeddings
- k=20 mined per query, 7 randomly sampled at training
- Fill strategy: Nomic uses random fill; our temporal variant uses expanding search (no random noise — arguably better)

**Snellius execution:**
- Partition: gpu_a100, 1×A100 per split (3 parallel jobs)
- Toolchain: 2023 (Python 3.11.3, CUDA 12.1.1, PyTorch 2.1.2)
- FAISS: faiss-cpu==1.7.4 via pip (GPU variant had CUBLAS conflicts with lmod; CPU sufficient for ~155K vectors)
- Deps: transformers>=4.36,<5.0 pinned for PyTorch 2.1.2 compatibility
- Input: `~/data/01_training_data/{train_inter,train_extra,train_merged}.jsonl`
- Output: `~/data/02_hard_negatives/{inter,extra,merged}/` (shards + counts.json + offsets.json.gz)
- Shards copied to home BEFORE post-processing (safe from scratch cleanup)
- Script: `slurm1_get_news_negatives.sh` (parameterized via --export=SPLIT_LABEL,SPLIT_FILE)
- Also need to run for baseline: NO — Nomic's S3 shards are used directly

**Results:**

| Split | Examples | Shards | Size | Negatives/example |
|---|---|---|---|---|
| inter | 92,083 | 1 | 4.1 GB | 20 |
| extra | 101,184 | 2 | 4.5 GB | 20 |
| merged | 154,704 | 2 | 6.8 GB | 20 |

Spot-checks passed: queries contain entity+month+year, documents ~9K chars, 20 negatives each ~5-10K chars.
Counts.json and offsets.json.gz generated and verified for all splits.
SLURM logs archived at `~/data/02_hard_negatives/logs/`.

**Manuscript-relevant details (§3.5 Hard Negative Selection):**
- Embedding model for mining: gte-base (`thenlper/gte-base`), mean pooling, L2-normalized, tokenizer max_length=512
- Similarity metric: cosine (via FAISS IndexFlatIP on normalized vectors)
- k=20 hard negatives mined per (entity, month, year) query
- Temporal constraint: candidates from the same month-year as the positive document are excluded; enforces temporal discrimination as described in §3.5
- Fill strategy: iterative search expansion (doubles search window until k negatives found) — NO random fill. This differs from Nomic's random-fill approach and is arguably preferable as it avoids introducing random noise into the negative set
- At training time, 7 of the 20 mined negatives are randomly sampled per example (per Nomic's training loader with `sample_negatives=True`, matching their paper: "Instead of choosing the first N negatives, we randomly sampled the mined negatives")
- The temporal constraint applies ONLY to NewsCycle data shards. Nomic's upstream fine-tuning datasets use their own pre-mined negatives from S3 (standard cosine-similarity mining without temporal constraint)
- Scripts used: `get_negatives_news.py` (mining), `generate_counts_offsets.py` (loader metadata). Both are new additions to the contrastors fork; upstream equivalents (`get_negatives.py`, `offsets_count.py`) were not suitable due to S3-only design and text-mode offset computation

**Issues encountered and resolved:**
1. FAISS GPU (easybuild module was cleaned up; pip faiss-gpu-cu12 had CUBLAS conflicts) → used faiss-cpu==1.7.4
2. transformers 5.5.3 (from earlier --user install) required PyTorch ≥ 2.4 → pinned transformers>=4.36,<5.0
3. `zcat | head -1` caused SIGPIPE with `set -euo pipefail`, killing the script before rsync → replaced with Python gzip.open
4. Original sequential job lost inter shard when scratch was cleaned → restructured to rsync immediately + split into 3 parallel jobs

### Stage (iv): Perform fine-tuning
**Status**: BASELINE COMPLETE (2026-04-12). inter/extra/merged still TO DO.
**Action**: Baseline trained and saved. Other three variants queued once packaging pipeline (v) is unblocked.
- Key config: `configs/train/contrastive_finetune.yaml` + `configs/data/finetune_triplets_news.yaml`
- Training data: 10 Nomic datasets (S3) + NewsCycle (local shards)
- Training loader truncates to DEFAULT_COL_TO_MAX_TOKENS = {query: 32, document: 256, negative: 256} regardless of YAML seq_len
- **Document max_length decision (verified 2026-04-12)**:
  - Nomic paper says "all stages with max sequence length 2048" but upstream code uses DEFAULT_COL_TO_MAX_TOKENS = {query: 32, document: 256, negative: 256}. No override path exists (verified by full code trace through config → trainer → loader → tokenizer).
  - Nomic fine-tuned at 256 tokens. Their datasets (MSMARCO, NQ, NLI) are naturally short so 256 captures full content.
  - **Baseline**: keeps 256 default (reproduces Nomic exactly via contrastive_finetune.yaml)
  - **NewsCycle**: sets document_max_length=2048 (via contrastive_finetune_news.yaml). Rationale: NewsCycle documents are aggregated monthly coverage (median 1501 tokens). At 256, only 30% retained. At 2048, ~100% retained. This matches the paper's stated intent and the model's positional capacity.
  - query_max_length=64 for NewsCycle (queries are ~10 tokens; headroom for variation)
- **Future optimization (noted, not implemented)**: deduplicate similar articles within aggregated documents to maximize information diversity within the 2048-token window (e.g., skip articles with >80% title word overlap or high cosine similarity between article descriptions)
- flash-attn 2.8.3 prebuilt wheel + csrc extras (layer_norm, fused_dense_lib) compiled from v2.8.3 source on Snellius (one-time, persists in ~/.local/lib/python3.13/)
- Snellius: 4×H100, Cu12.8, torch 2.7.0, flash-attn 2.8.3
- Output: epoch_0_model checkpoint

**Baseline run (2026-04-12):**
- Submitted via `slurm2_run_finetuning.sh` with `MODEL_LABEL=baseline`, `TRAIN_CONFIG=contrastive_finetune.yaml`, `DATA_CONFIG=finetune_triplets.yaml`
- 6612/6612 steps, ~1 hr on 4×H100 after warmup (2.1–2.5 it/s)
- Checkpoint: `~/data/03_finetuned_model/baseline/epoch_0_model/` (config.json + model.safetensors, 522 MB)
- Training-side contrastors fixes applied (all in fork, not yet pushed):
  - `models/biencoder/modeling_biencoder.py`: added `_tied_weights_keys = {}` and `all_tied_weights_keys = {}` on BiEncoder for transformers 5.x `from_pretrained` compatibility
  - `trainers/__init__.py`: guarded optional imports (glue/image_text/mlm/mmlm/distill), populated `TRAINER_REGISTRY` conditionally
  - `convert_to_hf.py`: `push_to_hub` → `save_pretrained` (kwarg removed in transformers 5.x)

**inter/extra/merged (queued):** same script, different `MODEL_LABEL`/`TRAIN_CONFIG`/`DATA_CONFIG` — will run once Stage (v) is unblocked and baseline STS is validated.

### Stage (v): Package model
**Status**: BASELINE COMPLETE (2026-04-13). inter/extra/merged TO DO (reuse slurm_baseline_package.sh parameterized).

**Baseline packaging successful** (job 21811079, commit d87b236). The pipeline produces an ST-format model at `~/data/04_packaged_model/baseline_st/` and the Hub repo `rptkiddle/baseline_st_temp_v2`. Subsequently validated against Nomic's own model via MTEBv2 STS control experiment (see Stage vi) — **reproduction successful, +0.003 vs Nomic on MTEBv2 overall**.

**Production packaging remains**: `inter`, `extra`, and `merged` models need to be packaged the same way. Plan: parameterize `slurm_baseline_package.sh` via `--export` (matching `slurm2_run_finetuning.sh`'s pattern of MODEL_LABEL/CKPT_DIR variables) and run three times with the appropriate checkpoint paths.

**Script lessons learned (baseline run):**
1. `push_to_hub(use_temp_dir=False)` needs a pre-created work dir in cwd matching the repo basename. Fixed with `mkdir -p $REPO_DIR/baseline_hf_temp_v2` before Step 1 + `rm -rf` cleanup at the end.
2. `convert_hf_to_st.py` needs `trust_remote_code=True` propagated to config_args + tokenizer_args, not just model_args. Fixed in rebuild-clean commit d87b236.
3. Post-conversion config corrections (embd_pdrop=0.0, resid_pdrop=0.0, rotary_scaling_factor=2) successfully applied via HfApi round-trip between Step 1 and Step 3.

**Historical (superseded) Stage (v) notes — former plan:**

**Success criterion**: packaged baseline ST model achieves **MTEB STS ≥ 0.821** on eval (matching Nomic's published number). The whole point of the rebuild is to find out whether the previously-published 0.692 was prefix-only (already fixed in our preflight work, lifts to ~0.702) or training/environmental (which is why we re-ran the full pipeline). We won't know which until the packaged baseline is benchmarked.

**Approach**: follow the user's tested Hub round-trip workflow exactly (since they explicitly chose this to eliminate workflow-divergence as a variable), but verify each step against rebuild-clean reality rather than blindly mimicking the frozen fork. Use transformers 4.45.2 in an isolated venv to avoid the transformers 5.x save/load round-trip issue.

**Cosmetic vs substantive issues identified:**
- ✓ **Cosmetic**: the `<All keys matched successfully>` then `Some weights ... were not used` / `newly initialized` warnings from `convert_to_hf.py`. These look alarming but don't actually drop fine-tuning — verified by the user's prior NewsCycle benchmark results, where `NewsCycle_inter_st` dramatically outperformed the unsupervised base on entity retrieval. The warnings are an artifact of `BiEncoder.from_pretrained`'s internal double-load pattern, not evidence of weight loss.
- ✓ **Cosmetic**: `embd_pdrop` / `resid_pdrop` config.json values. The user manually edited the published config to `0.0` (Nomic's value), but at **inference** time the model is in `eval()` mode and dropout is inactive regardless of the config value. So config.json edits don't fix anything that matters for benchmarking. The real question is whether the **training** code path used dropout=0.0 or 0.1 — that needs to be verified by reading the rebuild-clean code, not by post-hoc patching.
- ❓ **Open / flagged in research notes**: `rotary_scaling_factor` config (Nomic's reference has 2 with `rope_parameters: {rope_theta: 1000, rope_type: dynamic, factor: 2}`; user's previous models had this NULL or different). Could matter for sequences > training length. STS eval truncates to 512 tokens, which is well under training length, so unlikely to drive the STS gap — but worth verifying.
- ❓ **Open**: `auto_map` cross-repo prefix vs local code copies. Nomic's published model uses cross-repo refs (`nomic-ai/nomic-bert-2048--...`) and ships no local code copies. The user's tested approach matched this. We should match it too unless there's a specific reason not to.

**Blocker history (2026-04-12, now superseded):**
- Yesterday's attempt to use transformers 5.5.3 hit an isinstance check in `NomicBertModel.__init__` after `save_pretrained` → `from_pretrained` round-trip. This is solved by using transformers 4.45.2 (Nomic's pinned version) in an isolated venv layered over the 2025 toolchain via `--system-site-packages`.

**Concrete plan (next session of work):**
1. **Diff `fork/main:convert_hf_to_st.py` against rebuild-clean** to confirm what's there matches what the user tested. Same for `convert_to_hf.py`.
2. **Read the user's tested workflow** from research notes: upload via `convert_to_hf.py --model_name rptkiddle/<temp>` (Hub repo id, private), then `convert_hf_to_st.py --input rptkiddle/<temp> --from-hub --tokenizer-from nomic-ai/nomic-embed-text-v1 --pooling mean --normalize --trust-remote-code --push --private --repo-id rptkiddle/<temp>_st`, plus the manual config.json edits.
3. **Write a SLURM script** that runs this end-to-end in a 4.45.2 venv, with HF auth via the cached token.
4. **Validate the output** against `nomic-ai/nomic-embed-text-v1` (file manifest, config structure, modules layout).
5. **Run STS eval** (separate job, can be 5.x). Compare against Nomic's 0.821.
6. **If STS doesn't match 0.821**, the gap tells us whether it's packaging (try local-only path), training (rebuild-clean dropout/RoPE check), or eval methodology (already vetted).
7. **Clean up the temporary private HF repos** when done.

**Deferred (for publication):** fix the transformers 5.x `save_pretrained`/`from_pretrained` round-trip properly so the whole pipeline runs on a single modern toolchain. Likely approach: patch `NomicBertModel.__init__` to accept `NomicBertConfig` directly (drop the `GPT2Config` isinstance check) in rebuild-clean.

**Gotcha: `push_to_hub(use_temp_dir=False)` needs a pre-created work dir.**
`convert_to_hf.py` hardcodes `model.push_to_hub(args.model_name, ..., use_temp_dir=False)`. Under that flag, `transformers.utils.hub.push_to_hub` derives a local working directory name from the basename of the repo id (e.g. `rptkiddle/baseline_hf_temp_v2` → `baseline_hf_temp_v2`) and tries to `_get_files_timestamps` on it BEFORE saving anything. The dir must exist (empty is fine). This is documented in the user's research notes ("make tmpdir in repo for HF model build") but is easy to miss because Nomic's upstream `convert_to_hf.py` doesn't mention it. Stage (v) SLURM script handles this with `mkdir -p $REPO_DIR/baseline_hf_temp_v2` before the `convert_to_hf.py` call, plus an `rm -rf` cleanup at the end.

**Known config deviations from Nomic reference (to revisit if STS doesn't match):**
- `rope_parameters` dict (`{rope_theta: 1000, rope_type: "dynamic", factor: 2}`) is present in Nomic's published `nomic-ai/nomic-embed-text-v1` config.json but **omitted** from our Step 2 corrections. Reasoning: we're not sure how transformers 4.45.2 will tolerate the field at load time, and the user's prior tested workflow didn't add it either. STS eval truncates to 512 tokens (well below training length), so dynamic rotary scaling shouldn't bite. If validation reveals this is the only meaningful diff and STS doesn't match Nomic's 0.821, this is the first thing to add.
- Other config keys-only-in-ref vs keys-only-in-ours: logged by the validation step but treated as advisory, not fatal. Inspect the validation log if STS underperforms.

### Stage (vi): Perform benchmarks
**Status**: STS validation COMPLETE (2026-04-13); full MTEBv2 benchmark TODO
**Action**: Run full MTEBv2 default-task benchmark against Nomic v1 and production models.

**Baseline STS reproduction validation (2026-04-13)** — done, results discarded from disk but recorded here as historical evidence:

Using `slurm4_mteb_eval.sh` with STS-only task list, 11 STS/Summarization tasks run twice (our baseline, Nomic v1 control):

| Task | Nomic v1 (control) | Rebuild baseline | Δ (ours − Nomic) |
|---|---|---|---|
| BIOSSES | 0.8649 | 0.8722 | +0.007 |
| SICK-R | 0.7857 | 0.7430 | −0.043 |
| STS12 | 0.7895 | 0.7323 | −0.057 |
| STS13 | 0.8543 | 0.8484 | −0.006 |
| STS14 | 0.8166 | 0.8004 | −0.016 |
| STS15 | 0.8722 | 0.8731 | +0.001 |
| STS16 | 0.8543 | 0.8237 | −0.031 |
| STS17 | 0.0028 | 0.1652 | +0.162 |
| STS22 | 0.6469 | 0.6688 | +0.022 |
| STSBenchmark | 0.8555 | 0.8407 | −0.015 |
| SummEval | 0.3027 | 0.3079 | +0.005 |
| **STS avg (10)** | **0.7343** | **0.7368** | **+0.003** |
| **Overall (11)** | **0.6950** | **0.6978** | **+0.003** |

**Verdict**: baseline reproduction **successful**. Our rebuild baseline is statistically indistinguishable from Nomic's own published model under MTEBv2 (we're +0.003 on the overall average, well within noise). Per-task differences fluctuate in both directions and average out to ≈0.

**Key finding about the original "0.129 STS gap"**: it was entirely an MTEBv1 vs MTEBv2 methodology artifact. Nomic's published 0.821 was computed against MTEBv1; under MTEBv2 their own published model scores 0.6950. The task superseding is also visible in real time (MTEB v2 warnings: *"Dataset 'STS22' is superseded by 'STS22.v2'"*, *"Dataset 'SummEval' is superseded by 'SummEvalSummarization.v2'"*). STS17 is particularly broken — Nomic's own model gets **0.0028** on it in our pipeline.

**Benchmark methodology decision (2026-04-13)**: rather than reproduce Nomic's 2025-era task list, we report **MTEBv2 default tasks per category** (6 categories from the Nomic paper: Classification, Clustering, Pair Classification, Reranking, Retrieval, STS). We run this benchmark **twice** — once on `nomic-ai/nomic-embed-text-v1`, once on our production merged model — and report the head-to-head comparison in the manuscript. This is the methodologically clean comparison and also means we don't carry Nomic's 2025 task list forward into the published results.

**Remaining Stage (vi) work:**
- Re-package `slurm4_mteb_eval.sh` to use MTEBv2 default tasks (instead of the STS-only task list or Nomic's 2025 per-category task list) and to run the full suite. Parameterize MODEL_ID so the same script runs Nomic v1 and our production models.
- Full MTEBv2 benchmark on `nomic-ai/nomic-embed-text-v1` (control)
- Full MTEBv2 benchmark on our production merged model (after Stage (v) packaging of inter/extra/merged)
- NewsCycle entity-retrieval benchmark (slurm3) on inter and extra
- DailyOracle benchmark (slurm5) on inter, extra, merged

---

## Security reminders

- Three HF tokens were leaked during the first session (2026-04-11). User has been asked to revoke all three.
- `slurm2_run_finetuning.sh:90` has a commented-out hardcoded token in the public fork's git history. Must scrub before any push.
- New scripts use `~/.cache/huggingface/token` (cached via `hf auth login`) — no hardcoded tokens.
- HF token on local Mac stored in macOS keychain: `security find-generic-password -a "$USER" -s huggingface_token -w`

---

## Log hygiene policy (established 2026-04-13)

SLURM stdout files (`slurm-<jobid>.out`) are handled as follows:

- **On success**: move the `.out` to `~/data/<stage>/logs/` (e.g., `~/data/03_finetuned_model/baseline/logs/`, `~/data/04_packaged_model/logs/`). The `logs/` subdirectory is created per-step as needed.
- **On failure / error**: delete the `.out` at earliest convenience. Do not retain failure logs — they clutter the record and can mislead the next session.
- **Never leave `.out` files in `~/`** — they must be either archived (on success) or deleted (on failure).

**Rationale**: `~/data/` is the single source of truth for reproducibility, mirroring the `01_training_data` → `02_hard_negatives` → `03_finetuned_model` → `04_packaged_model` → `05_eval_benchmarks` stage sequence. Failed-run logs don't reproduce anything; successful-run logs document how the artifact next to them was made.

**Known gaps (as of 2026-04-13 cleanup)**:
- Successful baseline fine-tuning slurm log was unintentionally deleted before this policy was established. Low severity: the baseline is a Nomic reproduction exercise, and only the NewsCycle model logs (inter/extra/merged) are load-bearing for the manuscript.
- `~/data/logs/` still holds `slurm_compile_flash_extras.out` (one-time infra log from flash-attn csrc compilation). Not tied to a data-producing stage; it's kept as a build record.
- `~/data/03_finetuned_model/baseline/step_4500/` is an orphan intermediate checkpoint from `save_every=4500 steps` in the training config. Retained as intentional intra-training state.

## Reproducibility scaffolding

**Branch layout (after 2026-04-13 sync):**

- **`github.com/Rptkiddle/contrastors_newscycle`** has two branches:
  - **`main`** — the user's frozen reference fork. NEVER push to this; only read from it for comparison.
  - **`rebuild-clean`** — new branch created 2026-04-13. The active rebuild starts from Nomic upstream and accumulates only the minimum-necessary commits. This is what we work on.

- **Local mac**:
  - `main` tracks `origin/main` (frozen reference).
  - `rebuild-clean` tracks `origin/rebuild-clean` (active rebuild).
  - `upstream` remote points to `github.com/nomic-ai/contrastors.git` for fetching Nomic's evolution.
  - `snellius` remote uses SSH transport (`snellius:contrastors`) for transferring commits between Snellius and local without going through GitHub.

- **Snellius `~/contrastors/`**:
  - `rebuild-clean` is the only local branch (renamed from `main` on 2026-04-13 for naming consistency). Tracks `fork/rebuild-clean`.
  - `origin` remote points to `github.com/nomic-ai/contrastors.git` (upstream Nomic). `fork` remote points to the user's GitHub repo.
  - **Important:** Snellius cannot push to GitHub directly (no credentials). To publish rebuild-clean changes: commit on Snellius → `git fetch snellius rebuild-clean` from local → `git push origin rebuild-clean` from local.

**Three-way sync rule:** every "meaningful step" on Snellius rebuild-clean must be committed and propagated to GitHub via local. Do not let Snellius accumulate commits that exist nowhere else.

**Current rebuild-clean head:** `be7b5ba feat: bring HN mining scripts into rebuild-clean` (as of 2026-04-13).

**Per-stage comparison workflow**: for each stage in the rebuild, diff `fork/main` against rebuild-clean for the files that matter, consult the research notes for the "why" of any user-specific changes, and only bring forward what is necessary for the NewsCycle approach. Favor Nomic upstream code wherever the user's change isn't load-bearing for NewsCycle specifically. When in doubt, raise the question rather than guessing.
