import json
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_label_vocab(entity_types):
    labels = ["O"]
    for t in entity_types:
        labels += [f"B-{t}", f"I-{t}"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def extract_spans(tags):
    """BIO tags -> list of (start, end_inclusive, type)."""
    spans = []
    start = None
    cur_type = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, cur_type))
            start = i
            cur_type = tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:] and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1, cur_type))
            start = None
            cur_type = None
    return spans


def entity_level_f1(gold_tags_list, pred_tags_list):
    """Strict entity-level micro-F1, matching the paper's primary metric
    (Table 2: 'entity-level F1 with strict matching')."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_tags_list, pred_tags_list):
        gold_spans = set(extract_spans(gold))
        pred_spans = set(extract_spans(pred))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def partial_match_f1(gold_tags_list, pred_tags_list, overlap_threshold=0.5):
    """Partial-match F1 using a token-overlap threshold (Table 2 secondary
    metric): a predicted span counts as a match if it shares at least
    `overlap_threshold` fraction of tokens (by IoU) with a gold span of
    the same type."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_tags_list, pred_tags_list):
        gold_spans = extract_spans(gold)
        pred_spans = extract_spans(pred)
        matched_gold = set()
        matched_pred = set()
        for pi, (ps, pe, ptype) in enumerate(pred_spans):
            for gi, (gs, ge, gtype) in enumerate(gold_spans):
                if gi in matched_gold or ptype != gtype:
                    continue
                overlap = max(0, min(pe, ge) - max(ps, gs) + 1)
                union = max(pe, ge) - min(ps, gs) + 1
                iou = overlap / union if union else 0
                if iou >= overlap_threshold:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break
        tp += len(matched_pred)
        fp += len(pred_spans) - len(matched_pred)
        fn += len(gold_spans) - len(matched_gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
