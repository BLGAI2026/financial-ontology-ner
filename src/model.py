"""
Bundles the shared encoder and all four Section-4 components into one
module. train.py owns the actual multi-task training loop (how the
components' losses are combined per step); this class just wires the
sub-modules together and exposes the forward passes each one needs.
"""
import torch
import torch.nn as nn

from .encoder import MultilingualEncoder
from .confidence_filter import ConfidenceBasedFilter
from .multi_view_distillation import MultiViewDistillation
from .contrastive_alignment import TransformerProjectionHead
from .entity_attention import EntityAwareNERHead


class FinancialOntologyNER(nn.Module):
    def __init__(self, model_cfg, num_labels):
        super().__init__()
        self.encoder = MultilingualEncoder(
            model_cfg.encoder_name,
            use_pretrained=model_cfg.use_pretrained_encoder,
            dropout=model_cfg.dropout,
        )
        hidden_dim = self.encoder.hidden_size

        # Component 1
        self.confidence_filter = ConfidenceBasedFilter(
            hidden_dim=hidden_dim,
            mlp_hidden_sizes=tuple(model_cfg.confidence_hidden_sizes),
            dropout=model_cfg.dropout,
        )

        # Component 2
        self.distillation = MultiViewDistillation(
            ling_dim=hidden_dim,
            struct_hidden_dim=model_cfg.graphsage_hidden_dim,
            graphsage_layers=model_cfg.graphsage_layers,
            num_clusters=model_cfg.num_clusters,
        )

        # Component 3
        self.projection_head = TransformerProjectionHead(
            dim=model_cfg.graphsage_hidden_dim,
            num_layers=model_cfg.projection_layers,
            num_heads=model_cfg.projection_heads,
            dropout=model_cfg.dropout,
        )

        # Component 4 + NER classification
        self.ner_head = EntityAwareNERHead(
            token_dim=hidden_dim,
            ontology_dim=model_cfg.graphsage_hidden_dim,
            num_labels=num_labels,
            dropout=model_cfg.dropout,
        )

    def encode_texts(self, texts, device):
        return self.encoder(texts, device=device)

    def encode_tokens(self, input_ids, attention_mask):
        out = self.encoder.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def ner_forward(self, input_ids, attention_mask, ontology_nodes, labels=None):
        h_t = self.encode_tokens(input_ids, attention_mask)
        return self.ner_head(h_t, ontology_nodes, labels=labels, attention_mask=attention_mask)
