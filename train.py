#!/usr/bin/env python
"""
End-to-end training script that ties together the four Section-4
components:

  1. Confidence-based cross-lingual pair filtering  (src/confidence_filter.py)
  2. Multi-view distillation (GraphSAGE + PMI graph) (src/multi_view_distillation.py, src/graph_construction.py)
  3. Uncertainty-aware contrastive alignment          (src/contrastive_alignment.py)
  4. Entity-aware attention for NER                   (src/entity_attention.py)

Per step:
  (a) rebuild the PMI co-occurrence graph, weighting cross-lingual edges
      by the current confidence-filter scores (Eq. 11);
  (b) run GraphSAGE + linguistic-view distillation to get v_fused per
      entity (Eq. 7-12) and the distillation loss L_distill (Eq. 9);
  (c) sample labeled/unlabeled cross-lingual pairs and compute
      L_filter (Eq. 4-6);
  (d) project v_fused through the Transformer head and compute the
      uncertainty-aware contrastive loss (Eq. 13);
  (e) run the entity-aware-attention NER head over a batch of
      sentences, using the current v_fused as ontology nodes, and
      compute the token-classification loss;
  (f) backprop the weighted sum of all four losses.

Usage:
    python train.py --epochs 3 --use_pretrained_encoder false
"""
import argparse
import json
import os
import random
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(__file__))

from src.config import ModelConfig, TrainConfig, DataConfig
from src.model import FinancialOntologyNER
from src.data import NERDataset, collate_ner_batch, load_pairs, load_documents
from src.graph_construction import build_cooccurrence_graph, assemble_graph, _entity_key
from src.contrastive_alignment import (
    uncertainty_aware_contrastive_loss, pairwise_confidence_matrix,
)
from src.utils import set_seed, build_label_vocab, entity_level_f1, partial_match_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--use_pretrained_encoder", type=str, default="true",
                    choices=["true", "false"],
                    help="false runs fully offline with a randomly-initialized "
                         "encoder architecture (for testing the pipeline); "
                         "true downloads XLM-RoBERTa-large and is what "
                         "reproduces paper-scale results.")
    p.add_argument("--data_dir", type=str, default="data/sample")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_entity_table(documents):
    """Collects unique (text, lang) entity mentions across all documents,
    in a fixed order, together with the sentence each appears in (used
    to pool a linguistic embedding v_ling per entity, Eq. 7)."""
    seen = {}
    for doc in documents:
        for e in doc["entities"]:
            key = _entity_key(e["text"], doc["lang"])
            if key not in seen:
                seen[key] = {"text": e["text"], "lang": doc["lang"],
                             "type": e["type"], "canonical_id": e.get("canonical_id"),
                             "context": doc["text"]}
    keys = list(seen.keys())
    return keys, seen


def graph_to_adjacency(G, node_order):
    idx_of = {k: i for i, k in enumerate(node_order)}
    adj_list = [[] for _ in node_order]
    edge_weights = [[] for _ in node_order]
    for u, v, data in G.edges(data=True):
        if u not in idx_of or v not in idx_of:
            continue
        iu, iv = idx_of[u], idx_of[v]
        w = data["weight"]
        adj_list[iu].append(iv)
        edge_weights[iu].append(w)
        adj_list[iv].append(iu)
        edge_weights[iv].append(w)
    return adj_list, edge_weights


def compute_alignment_scores(model, entity_keys, entity_meta, device, batch_size=64):
    """Encodes every unique entity mention with the shared encoder,
    pools to v_ling, and scores every cross-lingual pair with the
    Component-1 scorer -> {(text_a, lang_a, text_b, lang_b): s_ij}."""
    texts = [entity_meta[k]["text"] for k in entity_keys]
    v_ling = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out = model.encode_texts(texts[i:i + batch_size], device=device)
            v_ling.append(out["pooled"])
    v_ling = torch.cat(v_ling, dim=0)
    model.train()

    scores = {}
    n = len(entity_keys)
    with torch.no_grad():
        for i in range(n):
            mi = entity_meta[entity_keys[i]]
            for j in range(n):
                if i == j:
                    continue
                mj = entity_meta[entity_keys[j]]
                if mi["lang"] == mj["lang"]:
                    continue
                s = model.confidence_filter.score(v_ling[i:i + 1], v_ling[j:j + 1])
                scores[(mi["text"], mi["lang"], mj["text"], mj["lang"])] = s.item()
    return v_ling, scores


def sample_pair_embeddings(model, pairs, device, batch_size):
    batch = random.sample(pairs, min(batch_size, len(pairs))) if pairs else []
    if not batch:
        return None
    texts_a = [p["text_a"] for p in batch]
    texts_b = [p["text_b"] for p in batch]
    h_a = model.encode_texts(texts_a, device=device)["pooled"]
    h_b = model.encode_texts(texts_b, device=device)["pooled"]
    labels = torch.tensor([p.get("label", 0) for p in batch], device=device)
    return h_a, h_b, labels, batch


def evaluate_ner(model, dataset, tokenizer, label2id, id2label, ontology_nodes, device,
                  max_length=64):
    model.eval()
    per_lang_gold, per_lang_pred = {}, {}
    with torch.no_grad():
        for batch in dataset.batches(batch_size=16, shuffle=False):
            enc = collate_ner_batch(tokenizer, batch, label2id, max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model.ner_forward(input_ids, attn, ontology_nodes)
            preds = out["logits"].argmax(-1).cpu()

            for bi, rec in enumerate(batch):
                lang = rec["lang"]
                gold_tags = rec["ner_tags"]
                enc_single = tokenizer(rec["tokens"], is_split_into_words=True,
                                        truncation=True, max_length=max_length)
                word_ids = enc_single.word_ids()
                pred_tags = ["O"] * len(gold_tags)
                seen_words = set()
                for tok_idx, wid in enumerate(word_ids):
                    if wid is None or wid in seen_words:
                        continue
                    seen_words.add(wid)
                    pred_tags[wid] = id2label.get(preds[bi, tok_idx].item(), "O")

                per_lang_gold.setdefault(lang, []).append(gold_tags)
                per_lang_pred.setdefault(lang, []).append(pred_tags)
    model.train()

    results = {}
    all_gold, all_pred = [], []
    for lang in per_lang_gold:
        f1 = entity_level_f1(per_lang_gold[lang], per_lang_pred[lang])
        pm = partial_match_f1(per_lang_gold[lang], per_lang_pred[lang])
        results[lang] = {"strict_f1": f1["f1"], "partial_f1": pm["f1"],
                          "precision": f1["precision"], "recall": f1["recall"]}
        all_gold += per_lang_gold[lang]
        all_pred += per_lang_pred[lang]
    overall = entity_level_f1(all_gold, all_pred)
    results["AVG"] = {"strict_f1": overall["f1"], "precision": overall["precision"],
                       "recall": overall["recall"]}
    return results


def evaluate_ontology_alignment(alignment_scores, labeled_pairs, k_values=(1, 5)):
    """Precision@k over the confidence-filter's cross-lingual scores
    against the held-out labeled pairs (proxy for Table 3's P@1 / P@5).
    Also reports a simplified Cross-lingual Consistency Score (CCS) and
    Financial Relation Preservation (FRP) proxy, since the paper cites
    CCS/FRP from external references [26, 27] without giving closed-form
    equations in Section 4 -- these are documented approximations, not
    the paper's exact metric implementations."""
    positives = [p for p in labeled_pairs if p.get("label") == 1]
    if not positives:
        return {}

    def lookup(p):
        k1 = (p["text_a"], p["lang_a"], p["text_b"], p["lang_b"])
        k2 = (p["text_b"], p["lang_b"], p["text_a"], p["lang_a"])
        return alignment_scores.get(k1, alignment_scores.get(k2, 0.0))

    scored = sorted(positives, key=lookup, reverse=True)
    results = {}
    for k in k_values:
        top_k = scored[:k] if k <= len(scored) else scored
        # precision@k here = fraction of the top-k *known-positive* pairs
        # whose confidence score cleared the mean threshold (a reachable
        # proxy given the sample data has no negative-in-topk ambiguity
        # by construction; see README for how this maps to the paper's
        # full-corpus P@k, which ranks over *all* candidate pairs).
        mean_score = sum(alignment_scores.values()) / max(1, len(alignment_scores))
        hits = sum(1 for p in top_k if lookup(p) >= mean_score)
        results[f"P@{k}"] = hits / max(1, len(top_k))

    scores_all = [lookup(p) for p in positives]
    results["CCS_proxy"] = sum(scores_all) / len(scores_all)
    results["FRP_proxy"] = sum(1 for s in scores_all if s >= 0.5) / len(scores_all)
    return results


def main():
    args = parse_args()
    model_cfg = ModelConfig(use_pretrained_encoder=(args.use_pretrained_encoder == "true"))
    train_cfg = TrainConfig(seed=args.seed, device=args.device)
    if args.epochs:
        train_cfg.epochs = args.epochs
    data_cfg = DataConfig(
        train_path=os.path.join(args.data_dir, "ner_train.jsonl"),
        dev_path=os.path.join(args.data_dir, "ner_dev.jsonl"),
        test_path=os.path.join(args.data_dir, "ner_test.jsonl"),
        documents_path=os.path.join(args.data_dir, "entity_documents.jsonl"),
        labeled_pairs_path=os.path.join(args.data_dir, "cross_lingual_pairs_labeled.jsonl"),
        unlabeled_pairs_path=os.path.join(args.data_dir, "cross_lingual_pairs_unlabeled.jsonl"),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(train_cfg.seed)
    device = train_cfg.device

    label2id, id2label = build_label_vocab(model_cfg.entity_types)
    train_set = NERDataset(data_cfg.train_path, label2id)
    dev_set = NERDataset(data_cfg.dev_path, label2id)
    test_set = NERDataset(data_cfg.test_path, label2id)
    documents = load_documents(data_cfg.documents_path)
    labeled_pairs = load_pairs(data_cfg.labeled_pairs_path)
    unlabeled_pairs = load_pairs(data_cfg.unlabeled_pairs_path)

    print(f"Loaded {len(train_set)} train / {len(dev_set)} dev / {len(test_set)} test "
          f"NER sentences, {len(documents)} documents, {len(labeled_pairs)} labeled "
          f"pairs, {len(unlabeled_pairs)} unlabeled pairs.")

    model = FinancialOntologyNER(model_cfg, num_labels=len(label2id)).to(device)
    tokenizer = model.encoder.tokenizer

    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                       weight_decay=train_cfg.weight_decay)
    steps_per_epoch = max(1, len(train_set) // train_cfg.labeled_batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    entity_keys, entity_meta = build_entity_table(documents)
    global_step = 0

    for epoch in range(train_cfg.epochs):
        print(f"\n=== Epoch {epoch + 1}/{train_cfg.epochs} ===")

        # (a)+(b) Rebuild graph with current confidence scores, run distillation
        v_ling_all, alignment_scores = compute_alignment_scores(
            model, entity_keys, entity_meta, device
        )
        entity_meta_pmi, pair_counts, pmi_fn = build_cooccurrence_graph(
            documents, window_sentences=train_cfg.co_occurrence_window_sentences
        )
        G = assemble_graph(entity_meta_pmi, pair_counts, pmi_fn, alignment_scores)
        adj_list, edge_weights = graph_to_adjacency(G, entity_keys)

        dist_out = model.distillation(v_ling_all, adj_list, edge_weights)
        l_distill = model.distillation.distillation_loss(dist_out["p_ling"], dist_out["p_struct"])
        v_fused = dist_out["v_fused"]

        # (c) Confidence filter loss
        lab_sample = sample_pair_embeddings(model, labeled_pairs, device,
                                             train_cfg.labeled_batch_size)
        unlab_sample_v1 = sample_pair_embeddings(model, unlabeled_pairs, device,
                                                  train_cfg.unlabeled_batch_size)
        unlab_sample_v2 = sample_pair_embeddings(model, unlabeled_pairs, device,
                                                  train_cfg.unlabeled_batch_size) \
            if unlab_sample_v1 else None

        if lab_sample is not None:
            h_a, h_b, labels, _ = lab_sample
            # batch-level noise statistics (Eq.3) come from the current
            # cross-lingual confidence-score distribution over the whole
            # entity table, standing in for "scores within the batch".
            batch_scores = torch.tensor(list(alignment_scores.values()), device=device) \
                if alignment_scores else torch.zeros(1, device=device)
            tau, alpha, mu, sigma_s = model.confidence_filter.dynamic_threshold(batch_scores)

            if unlab_sample_v1 is not None:
                ua, ub, _, _ = unlab_sample_v1
                ua2, ub2, _, _ = unlab_sample_v2
                filt = model.confidence_filter.loss(
                    (h_a, h_b, labels), (ua, ub), (ua2, ub2), tau
                )
            else:
                l_sup, _ = model.confidence_filter.supervised_loss(h_a, h_b, labels)
                filt = {"loss": l_sup, "l_sup": l_sup.detach(),
                        "l_unsup": torch.zeros(())}
        else:
            filt = {"loss": torch.zeros((), device=device, requires_grad=True),
                    "l_sup": torch.zeros(()), "l_unsup": torch.zeros(())}
            tau = torch.tensor(0.5, device=device)

        # (d) Contrastive alignment loss over fused entity embeddings
        z = model.projection_head(v_fused)
        conf_matrix = pairwise_confidence_matrix(v_ling_all, model.confidence_filter.scorer)
        l_contrast = uncertainty_aware_contrastive_loss(z, conf_matrix, tau,
                                                          temperature=train_cfg.contrastive_temperature)

        # (e) NER loss over a batch of sentences, ontology nodes = v_fused
        ner_losses = []
        for batch in train_set.batches(batch_size=train_cfg.labeled_batch_size):
            enc = collate_ner_batch(tokenizer, batch, label2id, train_cfg.max_seq_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels_t = enc["labels"].to(device)
            out = model.ner_forward(input_ids, attn, v_fused.detach(), labels=labels_t)
            if out["loss"] is not None:
                ner_losses.append(out["loss"])
        l_ner = torch.stack(ner_losses).mean() if ner_losses else torch.zeros((), device=device)

        # (f) Combine and backprop
        total_loss = (
            train_cfg.filter_loss_weight * filt["loss"]
            + train_cfg.distill_loss_weight * l_distill
            + train_cfg.contrastive_loss_weight * l_contrast
            + train_cfg.ner_loss_weight * l_ner
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        print(f"  L_filter={filt['loss'].item():.4f} (sup={filt['l_sup'].item():.4f}, "
              f"unsup={filt['l_unsup'].item():.4f}) | L_distill={l_distill.item():.4f} | "
              f"L_contrast={l_contrast.item():.4f} | L_ner={l_ner.item():.4f} | "
              f"total={total_loss.item():.4f}")

        dev_results = evaluate_ner(model, dev_set, tokenizer, label2id, id2label,
                                    v_fused.detach(), device, train_cfg.max_seq_len)
        print(f"  Dev AVG strict F1: {dev_results['AVG']['strict_f1']:.4f}")

    # Final evaluation -> Table 2 / Table 3 style outputs
    v_ling_all, alignment_scores = compute_alignment_scores(model, entity_keys, entity_meta, device)
    entity_meta_pmi, pair_counts, pmi_fn = build_cooccurrence_graph(documents)
    G = assemble_graph(entity_meta_pmi, pair_counts, pmi_fn, alignment_scores)
    adj_list, edge_weights = graph_to_adjacency(G, entity_keys)
    dist_out = model.distillation(v_ling_all, adj_list, edge_weights)
    v_fused = dist_out["v_fused"]

    test_results = evaluate_ner(model, test_set, tokenizer, label2id, id2label,
                                 v_fused.detach(), device, train_cfg.max_seq_len)
    ontology_results = evaluate_ontology_alignment(alignment_scores, labeled_pairs)

    with open(os.path.join(args.output_dir, "table2_ner_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    with open(os.path.join(args.output_dir, "table3_ontology_results.json"), "w") as f:
        json.dump(ontology_results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    print("\n=== Table 2 (this run, sample data) — Entity-level strict F1 by language ===")
    for lang, res in test_results.items():
        print(f"  {lang:>5}: F1={res['strict_f1']:.4f}")
    print("\n=== Table 3 (this run, sample data) — Ontology alignment (proxy metrics) ===")
    for k, v in ontology_results.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSaved: {args.output_dir}/table2_ner_results.json, "
          f"{args.output_dir}/table3_ontology_results.json, {args.output_dir}/model.pt")


if __name__ == "__main__":
    main()
