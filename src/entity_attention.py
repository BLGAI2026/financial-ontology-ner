"""
Component 4 — Entity-Aware Attention for NER Integration (Section 4.3, Eq. 14-15)

Implements:
  Eq.14  alpha_tk = softmax_k( h_t^T W_a n_k )
  Eq.15  h~_t     = h_t + sum_k alpha_tk * W_m n_k

where h_t is the contextual token representation from the base NER
encoder and {n_1,...,n_K} are ontology node embeddings (the fused
entity representations v_fused produced by Component 2). The enriched
representation h~_t feeds a standard linear-CRF-free BIO classification
head, fine-tuned jointly with the rest of the pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityAwareAttention(nn.Module):
    def __init__(self, token_dim, ontology_dim):
        super().__init__()
        self.W_a = nn.Linear(ontology_dim, token_dim, bias=False)   # scores h_t^T W_a n_k
        self.W_m = nn.Linear(ontology_dim, token_dim, bias=False)   # maps n_k into token space

    def forward(self, h_t, ontology_nodes):
        """
        h_t:             [B, L, D_tok]  contextual token representations
        ontology_nodes:   [K, D_onto]   ontology node embeddings (v_fused)
        returns:
            h_tilde:      [B, L, D_tok]
            attn_weights: [B, L, K]
        """
        if ontology_nodes.size(0) == 0:
            return h_t, torch.zeros(h_t.size(0), h_t.size(1), 0, device=h_t.device)

        keys = self.W_a(ontology_nodes)                 # [K, D_tok]
        scores = torch.einsum("bld,kd->blk", h_t, keys)  # h_t^T W_a n_k -- Eq.14
        attn = F.softmax(scores, dim=-1)                 # [B, L, K]

        values = self.W_m(ontology_nodes)                # [K, D_tok]
        context = torch.einsum("blk,kd->bld", attn, values)  # sum_k alpha_tk W_m n_k
        h_tilde = h_t + context                           # Eq.15
        return h_tilde, attn


class EntityAwareNERHead(nn.Module):
    """Enriches contextual token embeddings with ontology attention, then
    classifies each token into a BIO label."""

    def __init__(self, token_dim, ontology_dim, num_labels, dropout=0.1):
        super().__init__()
        self.attention = EntityAwareAttention(token_dim, ontology_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(token_dim, num_labels)

    def forward(self, h_t, ontology_nodes, labels=None, attention_mask=None):
        h_tilde, attn = self.attention(h_t, ontology_nodes)
        logits = self.classifier(self.dropout(h_tilde))       # [B, L, num_labels]

        loss = None
        if labels is not None:
            active = attention_mask.bool() if attention_mask is not None \
                else torch.ones_like(labels, dtype=torch.bool)
            loss = F.cross_entropy(
                logits[active], labels[active], ignore_index=-100
            )
        return {"logits": logits, "loss": loss, "attn": attn}
