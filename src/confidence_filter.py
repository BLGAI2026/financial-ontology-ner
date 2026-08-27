"""
Component 1 — Confidence-Based Cross-Lingual Pair Filtering (Section 4.1)

Implements:
  Eq.2  s_ij     = sigma(W_s . ReLU(W_f [h_i;h_j] + b_f) + b_s)
  Eq.3  tau      = mu - alpha * sigma_s          (dynamic per-batch threshold)
        alpha    = alpha0 + beta * tanh(W_a[mu;sigma_s] + b_a)   (dataset-adaptive gate)
  Eq.4  L_sup    = BCE over labeled pairs
  Eq.5  L_unsup  = consistency regularization between two stochastic
                   (dropout) forward passes of the same pair, pulled
                   toward the batch-adaptive threshold
  Eq.6  L_filter = L_sup + lambda * L_unsup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceScorer(nn.Module):
    """s_ij = sigma(W_s . ReLU(W_f [h_i;h_j] + b_f) + b_s)  -- Eq. 2"""

    def __init__(self, hidden_dim, mlp_hidden_sizes=(2048, 1024, 512), dropout=0.1):
        super().__init__()
        in_dim = 2 * hidden_dim
        layers = []
        prev = in_dim
        for h in mlp_hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=-1)
        logit = self.out(self.mlp(x)).squeeze(-1)
        return torch.sigmoid(logit), logit


class AdaptiveThresholdGate(nn.Module):
    """alpha = alpha0 + beta * tanh(W_a [mu; sigma_s] + b_a)   (Section 4.1, after Eq.3)

    Learns how strict the dynamic threshold tau = mu - alpha*sigma_s
    should be, conditioned on the batch-level noise statistics
    (mean/std of the pairwise confidence scores) rather than fixing
    alpha as a static hyperparameter.
    """

    def __init__(self, alpha0=1.5, beta=1.0):
        super().__init__()
        self.alpha0 = alpha0
        self.beta = beta
        self.gate = nn.Linear(2, 1)

    def forward(self, mu, sigma_s):
        stats = torch.stack([mu, sigma_s]).unsqueeze(0)
        alpha = self.alpha0 + self.beta * torch.tanh(self.gate(stats)).squeeze()
        tau = mu - alpha * sigma_s
        return tau, alpha


class ConfidenceBasedFilter(nn.Module):
    """Full Component 1: scorer + adaptive threshold + Eq.4-6 losses."""

    def __init__(self, hidden_dim, mlp_hidden_sizes=(2048, 1024, 512),
                 dropout=0.1, alpha0=1.5, beta=1.0, unsup_weight=0.3):
        super().__init__()
        self.scorer = ConfidenceScorer(hidden_dim, mlp_hidden_sizes, dropout)
        self.threshold_gate = AdaptiveThresholdGate(alpha0=alpha0, beta=beta)
        self.unsup_weight = unsup_weight

    def score(self, h_i, h_j):
        return self.scorer(h_i, h_j)[0]

    def dynamic_threshold(self, scores):
        mu = scores.mean()
        sigma_s = scores.std(unbiased=False) if scores.numel() > 1 else torch.zeros_like(mu)
        tau, alpha = self.threshold_gate(mu, sigma_s)
        return tau, alpha, mu, sigma_s

    def supervised_loss(self, h_i, h_j, labels):
        """Eq. 4: L_sup = BCE(s_ij, y_ij)."""
        scores, logits = self.scorer(h_i, h_j)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, scores

    def unsupervised_consistency_loss(self, h_i_v1, h_j_v1, h_i_v2, h_j_v2, tau):
        """Eq. 5: L_unsup = || s'_ij - s''_ij - delta(tau, s''_ij) ||_2^2

        h_*_v1 / h_*_v2 are two independently-dropout-perturbed encodings
        of the same pair (the paper's "two augmentations of the sample").
        delta(.) pulls the consistency target toward the current
        adaptive threshold, so scores near tau are treated as the most
        uncertain and penalized most for disagreeing.
        """
        s_prime, _ = self.scorer(h_i_v1, h_j_v1)
        s_double_prime, _ = self.scorer(h_i_v2, h_j_v2)
        delta = (tau - s_double_prime.detach())
        residual = s_prime - s_double_prime - delta
        loss = (residual ** 2).mean()
        return loss, s_prime, s_double_prime

    def loss(self, labeled_batch_embeds, unlabeled_batch_embeds_v1,
              unlabeled_batch_embeds_v2, tau):
        """Eq. 6: L_filter = L_sup + lambda * L_unsup"""
        h_i, h_j, labels = labeled_batch_embeds
        l_sup, sup_scores = self.supervised_loss(h_i, h_j, labels)

        ui1, uj1 = unlabeled_batch_embeds_v1
        ui2, uj2 = unlabeled_batch_embeds_v2
        l_unsup, s1, s2 = self.unsupervised_consistency_loss(ui1, uj1, ui2, uj2, tau)

        total = l_sup + self.unsup_weight * l_unsup
        return {
            "loss": total,
            "l_sup": l_sup.detach(),
            "l_unsup": l_unsup.detach(),
            "sup_scores": sup_scores.detach(),
        }
