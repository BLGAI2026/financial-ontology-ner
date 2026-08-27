import random

import torch

from .utils import read_jsonl


class NERDataset:
    def __init__(self, path, label2id):
        self.records = read_jsonl(path)
        self.label2id = label2id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def batches(self, batch_size, shuffle=True):
        idxs = list(range(len(self.records)))
        if shuffle:
            random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            yield [self.records[j] for j in idxs[i:i + batch_size]]


def align_labels_to_wordpieces(tokenizer, tokens, tags, label2id, max_length=64):
    """Tokenizes pre-split (word-level) tokens with `is_split_into_words`
    and propagates each word's BIO label to its first subword only
    (subsequent subwords get -100 / ignored in the loss)."""
    enc = tokenizer(
        tokens, is_split_into_words=True, truncation=True,
        max_length=max_length, padding="max_length", return_tensors="pt",
    )
    word_ids = enc.word_ids(batch_index=0)
    label_ids = []
    prev_word = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)
        elif wid != prev_word:
            label_ids.append(label2id.get(tags[wid], label2id["O"]))
        else:
            label_ids.append(-100)
        prev_word = wid
    return enc, torch.tensor(label_ids)


def collate_ner_batch(tokenizer, batch, label2id, max_length=64):
    input_ids, attn_masks, all_labels = [], [], []
    for rec in batch:
        enc, labels = align_labels_to_wordpieces(
            tokenizer, rec["tokens"], rec["ner_tags"], label2id, max_length
        )
        input_ids.append(enc["input_ids"][0])
        attn_masks.append(enc["attention_mask"][0])
        all_labels.append(labels)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn_masks),
        "labels": torch.stack(all_labels),
        "records": batch,
    }


def load_pairs(path):
    return read_jsonl(path)


def load_documents(path):
    return read_jsonl(path)
