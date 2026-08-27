"""
Central configuration. Default values are taken directly from the
paper's implementation-details paragraph (Section 5 / Results):

- Encoder: XLM-RoBERTa-large (550M params)
- Confidence scorer: 3-layer MLP, hidden sizes [2048, 1024, 512]
- GraphSAGE: 2 layers, mean aggregation, 768-d hidden states
- Projection head: 3-layer Transformer, 4 attention heads
- Optimizer: AdamW, lr=5e-5, linear warmup over 10% of steps
- Dropout: 0.1 everywhere
- Contrastive temperature T = 0.1
- alpha0 (filtering threshold init) = 1.5, unsupervised loss weight
  lambda = 0.3
- Batch sizes: 32 labeled / 128 unlabeled
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    encoder_name: str = "xlm-roberta-large"
    hidden_dim: int = 1024          # XLM-R-large hidden size
    use_pretrained_encoder: bool = True  # set False for offline/CI runs
    confidence_hidden_sizes: List[int] = field(default_factory=lambda: [2048, 1024, 512])
    graphsage_layers: int = 2
    graphsage_hidden_dim: int = 768
    projection_layers: int = 3
    projection_heads: int = 4
    dropout: float = 0.1
    num_clusters: int = 32          # K shared cluster centroids (Eq. 10)
    entity_types: List[str] = field(default_factory=lambda: ["ORG", "FIN", "MON", "IND"])


@dataclass
class TrainConfig:
    labeled_batch_size: int = 32
    unlabeled_batch_size: int = 128
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    epochs: int = 3
    grad_accum_steps: int = 4
    contrastive_temperature: float = 0.1
    distill_temperature: float = 0.1
    alpha0: float = 1.5             # filtering-threshold init (Eq. 3)
    beta_range: float = 1.0         # adaptation range for alpha gate
    unsup_loss_weight: float = 0.3  # lambda in Eq. 6
    filter_loss_weight: float = 1.0
    distill_loss_weight: float = 1.0
    contrastive_loss_weight: float = 1.0
    ner_loss_weight: float = 1.0
    seed: int = 42
    device: str = "cpu"
    max_seq_len: int = 64
    co_occurrence_window_sentences: int = 3


@dataclass
class DataConfig:
    train_path: str = "data/sample/ner_train.jsonl"
    dev_path: str = "data/sample/ner_dev.jsonl"
    test_path: str = "data/sample/ner_test.jsonl"
    documents_path: str = "data/sample/entity_documents.jsonl"
    labeled_pairs_path: str = "data/sample/cross_lingual_pairs_labeled.jsonl"
    unlabeled_pairs_path: str = "data/sample/cross_lingual_pairs_unlabeled.jsonl"
    languages: List[str] = field(default_factory=lambda: ["en", "zh", "es", "fr", "de", "ja"])
