"""
Component 3 — Uncertainty-Aware Contrastive Alignment (Section 4.3)

Implements Eq. 13:

    L_contrast = -1/|B| * sum_i log [
        sum_{j in P_i} w_ij * exp(sim(z_i, z_j) / T)
        --------------------------------------------------------------
        sum_{j in B} w_ij * exp(sim(z_i, z_j) / T)
          + sum_{k in N_i} exp(sim(z_i, z_k) / T)
    ]

where P_i = {j : s_ij >= tau} (positives from the confidence filter,
Section 4.1), w_ij = s_ij / sum_{p in P_i} s_ip normalizes the
confidence weights, and N_i are randomly sampled negatives.
z_i = Proj(v_fused) via a 3-layer Transformer projection head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerProjectionHead(nn.Module):
    """z_i = Proj(v_fused): 3-layer Transformer, 4 attention heads."""

    def __init__(self, dim, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, v_fused):
        # treat the batch of entity embeddings as a sequence of length 1
        # per entity so the projection can attend across the batch.
        x = v_fused.unsqueeze(0)                 # [1, N, dim] -> self-attention over the batch
        z = self.transformer(x).squeeze(0)        # [N, dim]
        return self.out_norm(z)


def uncertainty_aware_contrastive_loss(z, confidence_matrix, tau, temperature=0.1,
                                        num_negatives=8):
    """
    z:                  [N, dim] projected embeddings for the batch
    confidence_matrix:  [N, N] pairwise confidence scores s_ij (Eq. 2),
                         symmetric, diagonal ignored
    tau:                scalar dynamic threshold (Eq. 3) from the
                         confidence filter — defines positives P_i = {j: s_ij >= tau}
    temperature:        T in Eq. 13
    num_negatives:      |N_i|, sampled uniformly at random from non-positive,
                         non-self indices in the batch
    """
    n = z.size(0)
    device = z.device
    z_n = F.normalize(z, dim=-1)
    sim = z_n @ z_n.t() / temperature              # [N, N]
    exp_sim = torch.exp(sim)

    eye = torch.eye(n, dtype=torch.bool, device=device)
    is_positive = (confidence_matrix >= tau) & (~eye)

    losses = []
    for i in range(n):
        pos_idx = is_positive[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue  # no reliable positive for this anchor in this batch

        s_ip = confidence_matrix[i, pos_idx].clamp(min=1e-8)
        w_ip = s_ip / s_ip.sum()                       # normalized confidence weights
        numerator = (w_ip * exp_sim[i, pos_idx]).sum()

        # denominator term 1: sum over whole batch weighted by w_ij (using
        # confidence score directly as weight, 0 for non-positives per Eq.13's
        # w_ij definition which is only defined over P_i; we extend w_ij=0
        # outside P_i so the batch-sum term reduces to the positive sum).
        batch_weight = torch.zeros(n, device=device)
        batch_weight[pos_idx] = w_ip
        denom_pos_term = (batch_weight * exp_sim[i]).sum()

        # denominator term 2: randomly sampled negatives N_i
        non_positive = (~is_positive[i]) & (~eye[i])
        neg_candidates = non_positive.nonzero(as_tuple=True)[0]
        if neg_candidates.numel() > 0:
            k = min(num_negatives, neg_candidates.numel())
            sampled = neg_candidates[torch.randperm(neg_candidates.numel(), device=device)[:k]]
            denom_neg_term = exp_sim[i, sampled].sum()
        else:
            denom_neg_term = torch.zeros((), device=device)

        denom = denom_pos_term + denom_neg_term
        loss_i = -torch.log((numerator + 1e-12) / (denom + 1e-12))
        losses.append(loss_i)

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def pairwise_confidence_matrix(entities_h, confidence_scorer):
    """Materializes the [N,N] confidence-score matrix s_ij for a batch
    of entity embeddings using the Component-1 scorer (Eq. 2)."""
    n = entities_h.size(0)
    h_i = entities_h.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    h_j = entities_h.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    scores, _ = confidence_scorer(h_i, h_j)
    return scores.view(n, n)
