# Financial Ontology NER — Reference Implementation

Reference implementation of the four core components from Section 4 of
*"Utilization of advanced machine learning based on financial ontology
induction for adaptive multilingual method in document recognition"*:

1. **Confidence-based cross-lingual pair filtering** (Section 4.1, Eq. 2–6) — `src/confidence_filter.py`
2. **Multi-view distillation** — GraphSAGE structural view + PMI co-occurrence graph (Section 4.2, Eq. 7–12) — `src/multi_view_distillation.py`, `src/graph_construction.py`
3. **Uncertainty-aware contrastive alignment** (Section 4.3, Eq. 13) — `src/contrastive_alignment.py`
4. **Entity-aware attention** for NER integration (Section 4.3, Eq. 14–15) — `src/entity_attention.py`

`train.py` ties all four together into one multi-task training step; `evaluate.py`
reloads a checkpoint and reproduces the results tables without retraining.

## Repository layout

```
financial-ontology-ner/
├── README.md
├── requirements.txt
├── train.py                     # end-to-end training script (ties the 4 components together)
├── evaluate.py                  # reload a checkpoint, re-run Table 2 / Table 3 evaluation
├── scripts/
│   └── build_sample_data.py     # regenerates the sample dataset below
├── src/
│   ├── config.py                 # ModelConfig / TrainConfig / DataConfig (paper's hyperparameters)
│   ├── encoder.py                 # XLM-RoBERTa wrapper (Eq. 2, 7)
│   ├── confidence_filter.py       # Component 1 (Eq. 2–6)
│   ├── graph_construction.py      # PMI co-occurrence graph (Eq. 11)
│   ├── multi_view_distillation.py # Component 2: GraphSAGE + KL distillation + gated fusion (Eq. 7–12)
│   ├── contrastive_alignment.py   # Component 3 (Eq. 13)
│   ├── entity_attention.py        # Component 4 + NER head (Eq. 14–15)
│   ├── model.py                    # bundles encoder + all 4 components
│   ├── data.py                     # dataset / batching / label alignment
│   └── utils.py                    # seeding, BIO span extraction, entity-level F1
└── data/sample/                  # small real (non-placeholder) sample corpus, see below
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`torch==2.13.0` installs the CPU build by default on most platforms; if you
have a CUDA GPU, install the matching CUDA build of torch first
(see https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`
so pip doesn't downgrade it.

## About the sample data (important)

`data/sample/` is **not placeholder text** — every sentence is a real financial
statement, correctly tagged with `ORG` / `FIN` / `MON` / `IND` entity spans, in
all six languages used in the paper's evaluation (EN, ZH, ES, FR, DE, JA). It's
small on purpose (~40 sentences, ~40 unique entities) so the whole pipeline —
confidence filtering, graph construction, GraphSAGE, distillation, contrastive
alignment, and NER — runs end-to-end on a laptop CPU in under a minute.

**It is a pipeline-correctness fixture, not a benchmark.** It is far too small
to produce meaningful F1 scores or to reproduce the paper's numbers. To
reproduce Table 2 / Table 3, replace the files in `data/sample/` (same JSONL
schema, described below) with the real corpora:

- **MultiFin** — https://github.com/RasmusJoergensen/MultiFin (Jørgensen et al., 2023)
- **CLFE-10** — the 10-language parallel earnings-report corpus cited as [18] in the paper
- **BUSTER** — https://github.com/babelscape/buster (Zugarini et al., 2023)

and regenerate/replace:
- `ner_train.jsonl`, `ner_dev.jsonl`, `ner_test.jsonl` — tokenized sentences + BIO tags
- `entity_documents.jsonl` — documents with entity spans, for the PMI co-occurrence graph
- `cross_lingual_pairs_labeled.jsonl` — labeled cross-lingual entity pairs (confidence-filter supervision)
- `cross_lingual_pairs_unlabeled.jsonl` — unlabeled pairs (consistency regularization, Eq. 5)

Run `python scripts/build_sample_data.py` to see exactly how the sample files
are built from raw sentences, as a template for preprocessing the real corpora
(tokenization, entity span → BIO conversion, canonical-id-based cross-lingual
pair mining).

### JSONL schema

`ner_*.jsonl`:
```json
{"id": "en-train-0", "lang": "en", "text": "...", "tokens": ["Apple","Inc","."], "ner_tags": ["B-ORG","I-ORG","I-ORG"], "entities": [{"text": "Apple Inc.", "type": "ORG", "canonical_id": "apple_inc"}]}
```

`entity_documents.jsonl`:
```json
{"doc_id": "en-train-0", "lang": "en", "text": "...", "entities": [{"text": "...", "type": "ORG", "canonical_id": "apple_inc"}]}
```

`cross_lingual_pairs_labeled.jsonl` / `..._unlabeled.jsonl`:
```json
{"text_a": "Apple Inc.", "lang_a": "en", "text_b": "苹果公司", "lang_b": "zh", "label": 1}
```
(`label` omitted in the unlabeled file.)

## Running

### 1. Quick offline smoke test (no internet, no GPU, ~1 minute)

Uses a small randomly-initialized encoder instead of downloading XLM-R, so you
can verify the pipeline runs before committing to a full download/training run:

```bash
python train.py --epochs 3 --use_pretrained_encoder false --device cpu
```

### 2. Real training run (reproduces the paper's setup)

Requires internet access the first time, to download `xlm-roberta-large` from
the Hugging Face Hub:

```bash
python train.py \
  --epochs 3 \
  --use_pretrained_encoder true \
  --data_dir data/sample \
  --device cuda
```

Swap `--data_dir` to point at your prepared MultiFin/CLFE-10/BUSTER JSONL files
to reproduce Table 2 / Table 3 at paper scale. The paper's hyperparameters
(XLM-R-large, 3-layer MLP confidence scorer `[2048,1024,512]`, 2-layer
GraphSAGE with 768-d hidden states, 3-layer / 4-head Transformer projection,
AdamW lr=5e-5 with 10% linear warmup, dropout 0.1, contrastive temperature
T=0.1, α₀=1.5, λ=0.3, batch sizes 32 labeled / 128 unlabeled) are the defaults
in `src/config.py` — override via CLI flags or by editing the dataclasses.

Outputs land in `outputs/`:
- `outputs/model.pt` — trained weights
- `outputs/table2_ner_results.json` — per-language entity-level F1 (Table 2 format)
- `outputs/table3_ontology_results.json` — ontology alignment metrics (Table 3 format)

### 3. Evaluate a saved checkpoint without retraining

```bash
python evaluate.py --checkpoint outputs/model.pt --use_pretrained_encoder true
```

## What each component actually computes

| Component | File | Paper equations | What it does |
|---|---|---|---|
| Confidence-based filtering | `src/confidence_filter.py` | Eq. 2–6 | Scores every cross-lingual pair `s_ij = σ(MLP([h_i;h_j]))`; learns a batch-adaptive threshold `τ = μ − α·σ` via a gated network (`α` itself is learned, not fixed); trains with BCE on labeled pairs + a dropout-consistency regularizer on unlabeled pairs. |
| Multi-view distillation | `src/multi_view_distillation.py`, `src/graph_construction.py` | Eq. 7–12 | Builds a PMI-weighted entity co-occurrence graph (cross-lingual edges scaled by the Component-1 confidence score); runs a 2-layer mean-aggregator GraphSAGE for the structural view; aligns it with the linguistic (XLM-R) view via bidirectional KL divergence over shared cluster assignments; fuses both views with a learned gate. |
| Uncertainty-aware contrastive alignment | `src/contrastive_alignment.py` | Eq. 13 | Projects fused entity embeddings through a 3-layer/4-head Transformer; positives are pairs above the dynamic threshold `τ`, weighted by their normalized confidence `w_ij`; negatives are randomly sampled from the rest of the batch. |
| Entity-aware attention | `src/entity_attention.py` | Eq. 14–15 | For each NER token, attends over the fused ontology-node embeddings (`v_fused`) and adds the resulting context vector to the token's contextual representation before the BIO classification layer. |

## Reproducing the headline results table

**Table 2 — Entity-level F1 (mean ± 95% CI), low-resource setting (5% labeled data)**
*(paper's reported numbers, reproduced here as the reference target — see note below)*

| Method | EN | ZH | ES | FR | DE | JA | AVG |
|---|---|---|---|---|---|---|---|
| XLM-R NER | 72.3 ± 1.84 | 65.1 ± 2.13 | 68.7 ± 1.76 | 63.4 ± 2.21 | 66.2 ± 1.95 | 61.8 ± 2.37 | 66.2 ± 1.92 |
| mBERT-ADAPT | 74.6 ± 1.63 | 67.1 ± 1.98 | 71.2 ± 1.72 | 66.5 ± 2.04 | 68.1 ± 1.81 | 63.9 ± 2.15 | 68.6 ± 1.78 |
| CLOnto | 75.1 ± 1.57 | 68.4 ± 1.84 | 70.0 ± 1.69 | 67.2 ± 1.91 | 69.0 ± 1.72 | 64.7 ± 2.03 | 69.1 ± 1.69 |
| NoiseRobustNER | 76.8 ± 1.42 | 70.2 ± 1.66 | 73.5 ± 1.51 | 68.9 ± 1.72 | 71.7 ± 1.54 | 66.5 ± 1.86 | 71.3 ± 1.55 |
| ContraOnto | 77.5 ± 1.31 | 71.0 ± 1.58 | 74.1 ± 1.43 | 69.8 ± 1.63 | 72.4 ± 1.47 | 67.2 ± 1.79 | 72.0 ± 1.47 |
| **Proposed (this repo)** | **82.1 ± 0.92** | **76.3 ± 1.07** | **79.8 ± 0.96** | **75.2 ± 1.13** | **78.0 ± 1.01** | **73.1 ± 1.22** | **77.4 ± 0.98** |

**Table 3 — Cross-lingual ontology alignment (mean ± 95% CI)**

| Metric | CLOnto | ContraOnto | Proposed |
|---|---|---|---|
| P@1 | 72.1 ± 1.76 | 81.3 ± 1.42 | **89.4 ± 0.83** |
| P@5 | 85.2 ± 1.21 | 88.7 ± 1.08 | **93.6 ± 0.64** |
| CCS | 68.5 ± 1.93 | 75.2 ± 1.57 | **83.9 ± 0.91** |
| FRP | 71.3 ± 1.84 | 78.6 ± 1.49 | **86.2 ± 0.87** |

To actually reproduce these numbers: swap in the full MultiFin + CLFE-10 +
BUSTER corpora as described above, set `--use_pretrained_encoder true`, and
train for the full schedule (the paper trains on 4× A100 GPUs; a single GPU
will just take longer). `train.py`/`evaluate.py` write the *same* metrics
(`outputs/table2_ner_results.json`, `outputs/table3_ontology_results.json`)
computed on your run, in a directly comparable format.

**Honesty note on Table 3's metrics:** the paper defines P@k, CCS, and FRP by
citing external references ([26], [27]) rather than giving closed-form
equations in Section 4. `evaluate_ontology_alignment()` in `train.py`
implements P@k directly (precision of the confidence filter's ranked
cross-lingual pairs against known-positive alignments) and documented,
labeled **proxy** versions of CCS/FRP; swap in the exact metric
implementations from those references if you need bit-for-bit comparable
numbers.

## Notes on the offline/tiny-encoder fallback

`src/encoder.py` downloads `xlm-roberta-large` (550M params) from the Hugging
Face Hub by default. If that fails — no internet, no cached weights — it
falls back to a small randomly-initialized architecture purely so the
pipeline is exercisable in offline/CI environments (that's what
`--use_pretrained_encoder false` does, and what was used to smoke-test this
repository end-to-end before packaging). This fallback will **not** produce
meaningful NER or alignment quality — it exists to validate wiring and
tensor shapes, not accuracy. Always use `--use_pretrained_encoder true` with
network access for real experiments.

## License / data note

`data/sample/` is original text written for this repository; the real
datasets (MultiFin, CLFE-10, BUSTER) are third-party releases — check their
respective licenses before redistributing.
