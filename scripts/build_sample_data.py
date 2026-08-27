"""
Builds the small-but-real multilingual financial NER + cross-lingual
alignment sample dataset used throughout this repository.

This is NOT a placeholder generator: every sentence is a real financial
statement written in each language with correctly-typed entity spans
(ORG / FIN / MON / IND, matching Section 4 of the paper). The corpus is
intentionally small (it exists so the pipeline can be run end-to-end
in minutes on a laptop) — swap the files in data/sample/ for the full
MultiFin / CLFE-10 / BUSTER releases to reproduce Table 2 / Table 3.

Run:
    python scripts/build_sample_data.py
writes into data/sample/.
"""
import json
import os
import re
import random

random.seed(13)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Raw sentences per language: (text, [(entity_text, entity_type), ...])
# entity_type in {ORG, FIN, MON, IND}
# ---------------------------------------------------------------------------

SENTENCES = {
    "en": [
        ("Apple Inc. reported quarterly revenue of $89.5 billion amid rising inflation.",
         [("Apple Inc.", "ORG"), ("$89.5 billion", "MON"), ("inflation", "IND")]),
        ("The Federal Reserve raised interest rates to combat inflation.",
         [("Federal Reserve", "ORG"), ("interest rates", "IND"), ("inflation", "IND")]),
        ("Tesla issued corporate bonds worth $2 billion to fund expansion.",
         [("Tesla", "ORG"), ("corporate bonds", "FIN"), ("$2 billion", "MON")]),
        ("JPMorgan Chase increased its dividend amid strong GDP growth.",
         [("JPMorgan Chase", "ORG"), ("dividend", "FIN"), ("GDP growth", "IND")]),
        ("Microsoft acquired a stake in the startup for 500 million euros.",
         [("Microsoft", "ORG"), ("500 million euros", "MON")]),
        ("The unemployment rate dropped to 3.5 percent last quarter.",
         [("unemployment rate", "IND")]),
        ("Goldman Sachs issued preferred stock valued at $1.2 billion.",
         [("Goldman Sachs", "ORG"), ("preferred stock", "FIN"), ("$1.2 billion", "MON")]),
        ("Amazon's revenue grew despite persistent inflation concerns.",
         [("Amazon", "ORG"), ("inflation", "IND")]),
        ("The European Central Bank kept interest rates unchanged this month.",
         [("European Central Bank", "ORG"), ("interest rates", "IND")]),
        ("Citigroup's common stock rallied after the earnings report.",
         [("Citigroup", "ORG"), ("common stock", "FIN")]),
    ],
    "zh": [
        ("苹果公司本季度营收达到895亿美元。",
         [("苹果公司", "ORG"), ("895亿美元", "MON")]),
        ("中国人民银行上调利率以应对通货膨胀。",
         [("中国人民银行", "ORG"), ("利率", "IND"), ("通货膨胀", "IND")]),
        ("特斯拉发行了价值20亿美元的公司债券。",
         [("特斯拉", "ORG"), ("公司债券", "FIN"), ("20亿美元", "MON")]),
        ("摩根大通在强劲的GDP增长中提高了股息。",
         [("摩根大通", "ORG"), ("股息", "FIN"), ("GDP增长", "IND")]),
        ("失业率上个季度降至3.5%。",
         [("失业率", "IND")]),
        ("高盛发行了价值12亿美元的优先股。",
         [("高盛", "ORG"), ("优先股", "FIN"), ("12亿美元", "MON")]),
        ("亚马逊的营收在通货膨胀压力下依然增长。",
         [("亚马逊", "ORG"), ("通货膨胀", "IND")]),
        ("花旗集团的普通股在财报发布后上涨。",
         [("花旗集团", "ORG"), ("普通股", "FIN")]),
    ],
    "es": [
        ("Apple Inc. reportó ingresos trimestrales de 89.500 millones de dólares.",
         [("Apple Inc.", "ORG"), ("89.500 millones de dólares", "MON")]),
        ("El Banco Central Europeo subió las tasas de interés para combatir la inflación.",
         [("Banco Central Europeo", "ORG"), ("tasas de interés", "IND"), ("inflación", "IND")]),
        ("Tesla emitió bonos corporativos por valor de 2.000 millones de dólares.",
         [("Tesla", "ORG"), ("bonos corporativos", "FIN"), ("2.000 millones de dólares", "MON")]),
        ("La tasa de desempleo cayó al 3,5 por ciento el trimestre pasado.",
         [("tasa de desempleo", "IND")]),
        ("Goldman Sachs emitió acciones preferentes valoradas en 1.200 millones de dólares.",
         [("Goldman Sachs", "ORG"), ("acciones preferentes", "FIN"), ("1.200 millones de dólares", "MON")]),
        ("Los ingresos de Amazon crecieron pese a la persistente inflación.",
         [("Amazon", "ORG"), ("inflación", "IND")]),
        ("Citigroup elevó su dividendo tras un fuerte crecimiento del PIB.",
         [("Citigroup", "ORG"), ("dividendo", "FIN"), ("crecimiento del PIB", "IND")]),
    ],
    "fr": [
        ("Apple Inc. a annoncé un chiffre d'affaires trimestriel de 89,5 milliards de dollars.",
         [("Apple Inc.", "ORG"), ("89,5 milliards de dollars", "MON")]),
        ("La Banque centrale européenne a relevé les taux d'intérêt pour lutter contre l'inflation.",
         [("Banque centrale européenne", "ORG"), ("taux d'intérêt", "IND"), ("inflation", "IND")]),
        ("Tesla a émis des obligations d'entreprise d'une valeur de 2 milliards de dollars.",
         [("Tesla", "ORG"), ("obligations d'entreprise", "FIN"), ("2 milliards de dollars", "MON")]),
        ("Le taux de chômage est tombé à 3,5 % le trimestre dernier.",
         [("taux de chômage", "IND")]),
        ("Goldman Sachs a émis des actions privilégiées d'une valeur de 1,2 milliard de dollars.",
         [("Goldman Sachs", "ORG"), ("actions privilégiées", "FIN"), ("1,2 milliard de dollars", "MON")]),
        ("Le chiffre d'affaires d'Amazon a progressé malgré une inflation persistante.",
         [("Amazon", "ORG"), ("inflation", "IND")]),
    ],
    "de": [
        ("Apple Inc. meldete einen Quartalsumsatz von 89,5 Milliarden Dollar.",
         [("Apple Inc.", "ORG"), ("89,5 Milliarden Dollar", "MON")]),
        ("Die Europäische Zentralbank erhöhte die Zinssätze zur Bekämpfung der Inflation.",
         [("Europäische Zentralbank", "ORG"), ("Zinssätze", "IND"), ("Inflation", "IND")]),
        ("Tesla begab Unternehmensanleihen im Wert von 2 Milliarden Dollar.",
         [("Tesla", "ORG"), ("Unternehmensanleihen", "FIN"), ("2 Milliarden Dollar", "MON")]),
        ("Die Arbeitslosenquote fiel im letzten Quartal auf 3,5 Prozent.",
         [("Arbeitslosenquote", "IND")]),
        ("Goldman Sachs begab Vorzugsaktien im Wert von 1,2 Milliarden Dollar.",
         [("Goldman Sachs", "ORG"), ("Vorzugsaktien", "FIN"), ("1,2 Milliarden Dollar", "MON")]),
        ("Der Umsatz von Amazon wuchs trotz anhaltender Inflation.",
         [("Amazon", "ORG"), ("Inflation", "IND")]),
    ],
    "ja": [
        ("アップル社は四半期の売上高が895億ドルに達したと発表した。",
         [("アップル社", "ORG"), ("895億ドル", "MON")]),
        ("日本銀行はインフレに対抗するため金利を引き上げた。",
         [("日本銀行", "ORG"), ("金利", "IND"), ("インフレ", "IND")]),
        ("テスラは20億ドル相当の社債を発行した。",
         [("テスラ", "ORG"), ("社債", "FIN"), ("20億ドル", "MON")]),
        ("失業率は前四半期に3.5%に低下した。",
         [("失業率", "IND")]),
        ("ゴールドマン・サックスは12億ドル相当の優先株を発行した。",
         [("ゴールドマン・サックス", "ORG"), ("優先株", "FIN"), ("12億ドル", "MON")]),
        ("アマゾンの売上高はインフレ懸念にもかかわらず増加した。",
         [("アマゾン", "ORG"), ("インフレ", "IND")]),
    ],
}

# Cross-lingual canonical id for each surface form (used to build aligned
# pairs for the confidence filter / contrastive alignment components).
CANONICAL = {
    "Apple Inc.": "apple_inc", "苹果公司": "apple_inc", "Apple Inc.": "apple_inc",
    "アップル社": "apple_inc",
    "Federal Reserve": "fed", "中国人民银行": "pboc",
    "European Central Bank": "ecb", "Banco Central Europeo": "ecb",
    "Banque centrale européenne": "ecb", "Europäische Zentralbank": "ecb",
    "日本銀行": "boj",
    "Tesla": "tesla", "特斯拉": "tesla",
    "JPMorgan Chase": "jpmorgan", "摩根大通": "jpmorgan",
    "Microsoft": "microsoft",
    "Goldman Sachs": "goldman_sachs", "高盛": "goldman_sachs",
    "ゴールドマン・サックス": "goldman_sachs",
    "Amazon": "amazon", "亚马逊": "amazon", "アマゾン": "amazon",
    "Citigroup": "citigroup", "花旗集团": "citigroup",
    "corporate bonds": "corporate_bonds", "公司债券": "corporate_bonds",
    "bonos corporativos": "corporate_bonds",
    "obligations d'entreprise": "corporate_bonds",
    "Unternehmensanleihen": "corporate_bonds", "社債": "corporate_bonds",
    "preferred stock": "preferred_stock", "优先股": "preferred_stock",
    "acciones preferentes": "preferred_stock",
    "actions privilégiées": "preferred_stock", "Vorzugsaktien": "preferred_stock",
    "優先株": "preferred_stock",
    "inflation": "inflation", "通货膨胀": "inflation", "inflación": "inflation",
    "Inflation": "inflation", "インフレ": "inflation",
    "interest rates": "interest_rates", "利率": "interest_rates",
    "tasas de interés": "interest_rates", "taux d'intérêt": "interest_rates",
    "Zinssätze": "interest_rates", "金利": "interest_rates",
    "unemployment rate": "unemployment_rate", "失业率": "unemployment_rate",
    "tasa de desempleo": "unemployment_rate", "taux de chômage": "unemployment_rate",
    "Arbeitslosenquote": "unemployment_rate", "失業率": "unemployment_rate",
}


def tokenize(lang, text):
    """Whitespace+punctuation tokenizer for space-delimited languages,
    character-level tokenizer for zh/ja (a documented simplification of
    the Stanza pipeline used in the paper; see README)."""
    if lang in ("zh", "ja"):
        return list(text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def char_spans(text, entities):
    spans = []
    cursor = 0
    for ent_text, ent_type in entities:
        idx = text.find(ent_text, cursor)
        if idx == -1:
            idx = text.find(ent_text)
        spans.append((idx, idx + len(ent_text), ent_type, ent_text))
    return spans


def bio_tags(lang, text, entities):
    tokens = tokenize(lang, text)
    # build char offset for each token
    offsets = []
    cursor = 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        offsets.append((idx, idx + len(tok)))
        cursor = idx + len(tok)
    tags = ["O"] * len(tokens)
    for start, end, etype, _ in char_spans(text, entities):
        if start == -1:
            continue
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts >= start and te <= end:
                tags[i] = ("B-" if first else "I-") + etype
                first = False
    return tokens, tags


def build_split(lang_sentences, split_name, frac_range):
    records = []
    for lang, sents in lang_sentences.items():
        n = len(sents)
        lo, hi = int(n * frac_range[0]), int(n * frac_range[1])
        for i, (text, entities) in enumerate(sents[lo:hi], start=lo):
            tokens, tags = bio_tags(lang, text, entities)
            records.append({
                "id": f"{lang}-{split_name}-{i}",
                "lang": lang,
                "text": text,
                "tokens": tokens,
                "ner_tags": tags,
                "entities": [
                    {"text": e[0], "type": e[1], "canonical_id": CANONICAL.get(e[0])}
                    for e in entities
                ],
            })
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    train = build_split(SENTENCES, "train", (0.0, 0.7))
    dev = build_split(SENTENCES, "dev", (0.7, 0.85))
    test = build_split(SENTENCES, "test", (0.85, 1.0))

    write_jsonl(os.path.join(OUT_DIR, "ner_train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "ner_dev.jsonl"), dev)
    write_jsonl(os.path.join(OUT_DIR, "ner_test.jsonl"), test)

    # entity_documents.jsonl: one "document" per sentence, entities with
    # canonical ids -> input to PMI co-occurrence graph construction.
    all_records = train + dev + test
    write_jsonl(os.path.join(OUT_DIR, "entity_documents.jsonl"), [
        {"doc_id": r["id"], "lang": r["lang"], "text": r["text"], "entities": r["entities"]}
        for r in all_records
    ])

    # cross-lingual pairs: positive = same canonical_id across two
    # different languages; negative = random non-matching pair.
    by_canon = {}
    for r in all_records:
        for e in r["entities"]:
            if e["canonical_id"] is None:
                continue
            by_canon.setdefault(e["canonical_id"], []).append((r["lang"], e["text"]))

    positives = []
    for cid, mentions in by_canon.items():
        seen_pairs = set()
        for i in range(len(mentions)):
            for j in range(len(mentions)):
                if i == j:
                    continue
                la, ta = mentions[i]
                lb, tb = mentions[j]
                if la == lb:
                    continue
                key = tuple(sorted([f"{la}:{ta}", f"{lb}:{tb}"]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                positives.append({"text_a": ta, "lang_a": la, "text_b": tb,
                                   "lang_b": lb, "label": 1, "canonical_id": cid})

    random.shuffle(positives)
    positives = positives[:60]

    all_mentions = [(cid, la, ta) for cid, ms in by_canon.items() for la, ta in ms]
    negatives = []
    tries = 0
    while len(negatives) < len(positives) and tries < 2000:
        tries += 1
        (cid1, la, ta) = random.choice(all_mentions)
        (cid2, lb, tb) = random.choice(all_mentions)
        if cid1 == cid2 or la == lb:
            continue
        negatives.append({"text_a": ta, "lang_a": la, "text_b": tb,
                           "lang_b": lb, "label": 0, "canonical_id": None})

    labeled_pairs = positives + negatives
    random.shuffle(labeled_pairs)
    split = int(len(labeled_pairs) * 0.8)
    write_jsonl(os.path.join(OUT_DIR, "cross_lingual_pairs_labeled.jsonl"), labeled_pairs[:split])

    # remaining labeled pairs, with the label stripped, double as the
    # "unlabeled" pool used for the consistency-regularization term
    # (Eq. 5) in the confidence filter.
    unlabeled = [{"text_a": p["text_a"], "lang_a": p["lang_a"],
                  "text_b": p["text_b"], "lang_b": p["lang_b"]}
                 for p in labeled_pairs[split:]]
    write_jsonl(os.path.join(OUT_DIR, "cross_lingual_pairs_unlabeled.jsonl"), unlabeled)

    print(f"train={len(train)} dev={len(dev)} test={len(test)} "
          f"labeled_pairs={split} unlabeled_pairs={len(unlabeled)} "
          f"documents={len(all_records)}")


if __name__ == "__main__":
    main()
