#!/usr/bin/env python
"""
Loads a checkpoint saved by train.py and re-runs the final Table 2 /
Table 3 style evaluation on the test split, without retraining.

Usage:
    python evaluate.py --checkpoint outputs/model.pt --use_pretrained_encoder false
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config import ModelConfig, TrainConfig, DataConfig
from src.model import FinancialOntologyNER
from src.data import NERDataset, load_pairs, load_documents
from src.graph_construction import build_cooccurrence_graph, assemble_graph
from src.utils import set_seed, build_label_vocab

from train import (
    build_entity_table, graph_to_adjacency, compute_alignment_scores,
    evaluate_ner, evaluate_ontology_alignment,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/model.pt")
    p.add_argument("--use_pretrained_encoder", type=str, default="true",
                    choices=["true", "false"])
    p.add_argument("--data_dir", type=str, default="data/sample")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    model_cfg = ModelConfig(use_pretrained_encoder=(args.use_pretrained_encoder == "true"))
    train_cfg = TrainConfig(device=device)
    data_cfg = DataConfig(
        test_path=os.path.join(args.data_dir, "ner_test.jsonl"),
        documents_path=os.path.join(args.data_dir, "entity_documents.jsonl"),
        labeled_pairs_path=os.path.join(args.data_dir, "cross_lingual_pairs_labeled.jsonl"),
    )

    label2id, id2label = build_label_vocab(model_cfg.entity_types)
    test_set = NERDataset(data_cfg.test_path, label2id)
    documents = load_documents(data_cfg.documents_path)
    labeled_pairs = load_pairs(data_cfg.labeled_pairs_path)

    model = FinancialOntologyNER(model_cfg, num_labels=len(label2id)).to(device)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint {args.checkpoint} not found — evaluating an "
              f"untrained model. Run train.py first.")
    tokenizer = model.encoder.tokenizer

    entity_keys, entity_meta = build_entity_table(documents)
    v_ling_all, alignment_scores = compute_alignment_scores(model, entity_keys, entity_meta, device)
    entity_meta_pmi, pair_counts, pmi_fn = build_cooccurrence_graph(documents)
    G = assemble_graph(entity_meta_pmi, pair_counts, pmi_fn, alignment_scores)
    adj_list, edge_weights = graph_to_adjacency(G, entity_keys)
    dist_out = model.distillation(v_ling_all, adj_list, edge_weights)
    v_fused = dist_out["v_fused"]

    test_results = evaluate_ner(model, test_set, tokenizer, label2id, id2label,
                                 v_fused.detach(), device, train_cfg.max_seq_len)
    ontology_results = evaluate_ontology_alignment(alignment_scores, labeled_pairs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "table2_ner_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    with open(os.path.join(args.output_dir, "table3_ontology_results.json"), "w") as f:
        json.dump(ontology_results, f, indent=2)

    print("\n=== Table 2 — Entity-level strict F1 by language ===")
    for lang, res in test_results.items():
        print(f"  {lang:>5}: F1={res['strict_f1']:.4f}")
    print("\n=== Table 3 — Ontology alignment (proxy metrics) ===")
    for k, v in ontology_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
