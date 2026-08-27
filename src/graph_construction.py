"""
Builds the entity co-occurrence graph G = (E, R) used by GraphSAGE
(Section 4.2). Two entities are connected whenever their mentions
co-occur within a window of 3 sentences (or the same document for
short filings). Edges are weighted by PMI, Laplace-smoothed for
low-frequency pairs, and cross-lingual edges are additionally scaled
by the confidence-filter alignment score s_ij (Eq. 11):

    w_ij = max(0, PMI(e_i, e_j)) * s_ij      (cross-lingual edges)
    w_ij = max(0, PMI(e_i, e_j))             (monolingual edges)
"""
import math
from collections import defaultdict
import networkx as nx


def _entity_key(text, lang):
    return f"{lang}::{text}"


def build_cooccurrence_graph(documents, window_sentences=3):
    """documents: list of {"doc_id", "lang", "text", "entities":[{"text","type",...}]}

    Since each sample "document" here is a single sentence, the
    3-sentence window collapses to co-occurrence within the same
    document, matching the paper's fallback rule for short financial
    filings ("or within the same document for shorter financial
    filings where sentence-level windows would be overly sparse").
    """
    pair_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    total_pairs = 0
    entity_meta = {}

    for doc in documents:
        lang = doc["lang"]
        ents = list({(e["text"], e["type"]) for e in doc["entities"]})
        keys = []
        for text, etype in ents:
            k = _entity_key(text, lang)
            entity_meta[k] = {"text": text, "lang": lang, "type": etype}
            entity_counts[k] += 1
            keys.append(k)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair = tuple(sorted([keys[i], keys[j]]))
                pair_counts[pair] += 1
                total_pairs += 1

    total_entities = sum(entity_counts.values())

    def pmi(k1, k2):
        # Laplace (add-one) smoothed PMI, stabilizes low-frequency pairs
        # common in low-resource languages.
        p_ij = (pair_counts.get(tuple(sorted([k1, k2])), 0) + 1) / (total_pairs + len(pair_counts) + 1)
        p_i = (entity_counts[k1] + 1) / (total_entities + len(entity_counts) + 1)
        p_j = (entity_counts[k2] + 1) / (total_entities + len(entity_counts) + 1)
        return math.log(p_ij / (p_i * p_j) + 1e-12)

    return entity_meta, pair_counts, pmi


def assemble_graph(entity_meta, pair_counts, pmi_fn, alignment_scores=None):
    """alignment_scores: optional dict {(text_a, lang_a, text_b, lang_b): s_ij}
    from the confidence filter (Section 4.1), used to scale cross-lingual
    edge weights (Eq. 11). Monolingual edges use PMI alone.
    """
    alignment_scores = alignment_scores or {}
    G = nx.Graph()
    for k, meta in entity_meta.items():
        G.add_node(k, **meta)

    for (k1, k2), _count in pair_counts.items():
        m1, m2 = entity_meta[k1], entity_meta[k2]
        raw_pmi = max(0.0, pmi_fn(k1, k2))
        if raw_pmi == 0.0:
            continue
        if m1["lang"] != m2["lang"]:
            s_ij = _lookup_alignment(alignment_scores, m1, m2)
            weight = raw_pmi * s_ij
        else:
            weight = raw_pmi
        if weight > 0:
            G.add_edge(k1, k2, weight=weight, cross_lingual=(m1["lang"] != m2["lang"]))
    return G


def _lookup_alignment(alignment_scores, m1, m2):
    key1 = (m1["text"], m1["lang"], m2["text"], m2["lang"])
    key2 = (m2["text"], m2["lang"], m1["text"], m1["lang"])
    if key1 in alignment_scores:
        return alignment_scores[key1]
    if key2 in alignment_scores:
        return alignment_scores[key2]
    # No confidence score available for this pair -> treat as filtered
    # out (s_ij defaults to 0), consistent with "retained only for pairs
    # that satisfy the confidence-based filtering criterion".
    return 0.0
