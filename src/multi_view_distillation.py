"""
Component 2 — Multi-View Distillation for Ontology Refinement (Section 4.2)

Implements:
  Eq.7   v_ling   = pool(XLM-R(mentions of entity e))
  Eq.8   v_struct = GraphSAGE(e; G)                       (2-layer, mean aggregator)
  Eq.9   L_distill = sum_e [ KL(p_ling^e || p_struct^e) + KL(p_struct^e || p_ling^e) ]
  Eq.10  p_view^e(k) = softmax_k( v_view^T c_k / tau )     (shared cluster centroids)
  Eq.11  w_ij = max(0, PMI(e_i,e_j)) * s_ij                (graph_construction.py)
  Eq.12  v_fused = beta * v_ling + (1-beta) * v_struct, beta = sigma(W_f[v_ling;v_struct])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Mean-aggregator GraphSAGE layer: h_v' = ReLU(W . [h_v ; mean_{u in N(v)} w_uv * h_u])"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_feats, adj_list, edge_weights):
        """node_feats: [N, in_dim]
        adj_list: list of length N, adj_list[v] = list of neighbor indices
        edge_weights: list of length N, edge_weights[v] = list of weights
                      aligned with adj_list[v]
        """
        n = node_feats.size(0)
        agg = torch.zeros_like(node_feats)
        for v in range(n):
            neighbors = adj_list[v]
            if not neighbors:
                continue
            w = torch.tensor(edge_weights[v], dtype=node_feats.dtype, device=node_feats.device)
            w = w / (w.sum() + 1e-8)
            neigh_feats = node_feats[neighbors]  # [deg, in_dim]
            agg[v] = (w.unsqueeze(-1) * neigh_feats).sum(0)
        combined = torch.cat([node_feats, agg], dim=-1)
        return F.relu(self.linear(combined))


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE, mean aggregation, 768-d hidden states (Eq. 8)."""

    def __init__(self, in_dim, hidden_dim=768, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            GraphSAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

    def forward(self, node_feats, adj_list, edge_weights):
        h = node_feats
        for layer in self.layers:
            h = layer(h, adj_list, edge_weights)
        return h


class ClusterAssignment(nn.Module):
    """p_view^e(k) = softmax(v_view^T c_k / tau) with shared centroids -- Eq. 10"""

    def __init__(self, dim, num_clusters=32, temperature=0.1):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim) * 0.02)
        self.temperature = temperature

    def forward(self, v):
        logits = v @ self.centroids.t() / self.temperature
        return F.softmax(logits, dim=-1)


class GatedFusion(nn.Module):
    """v_fused = beta*v_ling + (1-beta)*v_struct, beta = sigma(W_f[v_ling;v_struct]) -- Eq.12"""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, v_ling, v_struct):
        beta = torch.sigmoid(self.gate(torch.cat([v_ling, v_struct], dim=-1)))
        return beta * v_ling + (1 - beta) * v_struct, beta


class MultiViewDistillation(nn.Module):
    def __init__(self, ling_dim, struct_hidden_dim=768, graphsage_layers=2,
                 num_clusters=32, temperature=0.1):
        super().__init__()
        self.graphsage = GraphSAGEEncoder(ling_dim, struct_hidden_dim, graphsage_layers)
        # project both views to a common dim for clustering/fusion
        common_dim = struct_hidden_dim
        self.ling_proj = nn.Linear(ling_dim, common_dim)
        self.cluster_ling = ClusterAssignment(common_dim, num_clusters, temperature)
        self.cluster_struct = ClusterAssignment(common_dim, num_clusters, temperature)
        self.fusion = GatedFusion(common_dim)

    def forward(self, v_ling_raw, adj_list, edge_weights):
        """v_ling_raw: [N, ling_dim] pooled XLM-R embeddings per entity (Eq.7)"""
        v_struct = self.graphsage(v_ling_raw, adj_list, edge_weights)   # Eq. 8
        v_ling = self.ling_proj(v_ling_raw)

        p_ling = self.cluster_ling(v_ling)      # Eq. 10
        p_struct = self.cluster_struct(v_struct)

        v_fused, beta = self.fusion(v_ling, v_struct)  # Eq. 12
        return {
            "v_ling": v_ling, "v_struct": v_struct, "v_fused": v_fused,
            "p_ling": p_ling, "p_struct": p_struct, "beta": beta,
        }

    @staticmethod
    def distillation_loss(p_ling, p_struct, eps=1e-8):
        """Eq. 9: bidirectional KL divergence between cluster-assignment
        distributions of the two views."""
        kl_1 = F.kl_div((p_ling + eps).log(), p_struct, reduction="batchmean")
        kl_2 = F.kl_div((p_struct + eps).log(), p_ling, reduction="batchmean")
        return kl_1 + kl_2
