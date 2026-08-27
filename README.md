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


