"""
gnn_model.py - GNN Delay Predictor with multi-head attention
                + Spatio-Temporal extension for dynamic satellite topologies
"""

import os
import gc
import math
import logging
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mixed precision availability
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Multi-head attention
# ===========================================================================
class MultiHeadAttention(nn.Module):
    """Standard multi-head self/cross attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, query, key, value,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = True):
        B = query.size(0)
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        out = self.out_proj(out)
        return (out, attn) if return_attention_weights else (out, None)


# ===========================================================================
# 2. Graph Attention layer
# ===========================================================================
class GraphAttentionLayer(nn.Module):
    """Single GAT layer used in the full RouteNet-Fermi model."""

    def __init__(self, in_features: int, out_features: int, num_heads: int,
                 dropout: float = 0.1, concat: bool = True,
                 edge_feat_dim: int = 9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(in_features, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_features),
        )
        self.edge_proj = nn.Linear(edge_feat_dim, in_features)
        self.residual = nn.Linear(in_features, out_features) \
            if in_features != out_features else nn.Identity()

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False):
        N = node_features.size(0)
        # Build dense attention mask from edge_index
        mask = torch.zeros((1, N, N), device=node_features.device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            mask[0, src, dst] = 1.0
        # Always allow self-attention
        mask[0, torch.arange(N), torch.arange(N)] = 1.0

        q = k = v = node_features.unsqueeze(0)
        out, attn = self.attention(q, k, v, mask, return_attention_weights)
        out = out.squeeze(0)

        if edge_features is not None and edge_features.size(0) > 0:
            # Aggregate edge features into nodes
            e = self.edge_proj(edge_features)
            # edge_features is per-undirected-edge; broadcast to both endpoints
            if e.size(0) * 2 == edge_index.size(1):
                e_full = torch.cat([e, e], dim=0)
            else:
                e_full = e[:edge_index.size(1)]
            agg = torch.zeros_like(out)
            agg.index_add_(0, edge_index[1], e_full[:edge_index.size(1)])
            out = out + agg

        out = self.ffn(out)
        out = out + self.residual(node_features)
        return out, attn


# ===========================================================================
# 2b. Spatio-Temporal Message-Passing Layer
# ===========================================================================
class SpatioTemporalLayer(nn.Module):
    """
    Single spatio-temporal message-passing layer.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 edge_dim_in: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ---- spatial ----
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        edge_dim_in = edge_dim_in if edge_dim_in is not None else hidden_dim
        self.edge_proj = nn.Linear(edge_dim_in, hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_dropout = nn.Dropout(dropout)

        # ---- temporal ----
        # GRUCell: input = spatial output, hidden = per-node memory
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
   
        N = node_features.size(0)
        device = node_features.device

        # ------------------------------------------------------------
        # (a) SPATIAL: multi-head attention over current edges
        # ------------------------------------------------------------
        mask = torch.zeros((1, N, N), device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            mask[0, src, dst] = 1.0
        idx = torch.arange(N, device=device)
        mask[0, idx, idx] = 1.0  # always allow self-attention

        q = k = v = node_features.unsqueeze(0)
        spatial_out, _ = self.attention(q, k, v, mask,
                                        return_attention_weights=False)
        spatial_out = spatial_out.squeeze(0)

        # Aggregate edge features into receiving nodes
        if edge_features is not None and edge_features.size(0) > 0:
            e = self.edge_proj(edge_features)
            # Edge features are stored per undirected edge; broadcast to
            # both directions if edge_index doubles them up.
            if e.size(0) * 2 == edge_index.size(1):
                e_full = torch.cat([e, e], dim=0)
            else:
                e_full = e[:edge_index.size(1)]
            agg = torch.zeros_like(spatial_out)
            agg.index_add_(0, edge_index[1], e_full[:edge_index.size(1)])
            spatial_out = spatial_out + agg

        # Spatial residual + norm + dropout
        spatial_out = self.spatial_norm(spatial_out + node_features)
        spatial_out = self.spatial_dropout(spatial_out)

        # ------------------------------------------------------------
        # (b) TEMPORAL: GRU cell over per-node hidden state
        # ------------------------------------------------------------
        if hidden_state is None \
                or hidden_state.size(0) != N \
                or hidden_state.size(1) != self.hidden_dim:
            hidden_state = torch.zeros(N, self.hidden_dim, device=device)

        new_hidden = self.gru(spatial_out, hidden_state)
        new_hidden = self.temporal_norm(new_hidden)
        return new_hidden


# ===========================================================================
# 3. Full RouteNet-Fermi-style model
# ===========================================================================
class RouteNetFermiWithMultiHeadAttention(nn.Module):
    """GNN model for predicting end-to-end delay."""

    def __init__(self,
                 node_feat_dim: int = 7,
                 edge_feat_dim: int = 9,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_mixed_precision: bool = True):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_precision = use_mixed_precision

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        # Pass `edge_feat_dim=hidden_dim` because the GAT receives the encoded edges
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads,
                                dropout, concat=(i < num_layers - 1),
                                edge_feat_dim=hidden_dim)
            for i in range(num_layers)
        ])

        # Heads (3 = mean + max + attn pool concat)
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.loss_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.throughput_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        # Per-flow head: concat(src, dst, global) → 1
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.global_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                flow_pairs: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        if self.use_mixed_precision and AMP_AVAILABLE and node_features.is_cuda:
            with autocast():
                return self._forward_impl(node_features, edge_index,
                                          edge_features, flow_pairs,
                                          return_attention_weights)
        return self._forward_impl(node_features, edge_index, edge_features,
                                  flow_pairs, return_attention_weights)

    def _forward_impl(self, node_features, edge_index, edge_features,
                      flow_pairs, return_attention_weights):
        x = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features) if edge_features is not None else None

        attn_list = []
        for layer in self.gat_layers:
            x, attn = layer(x, edge_index, e, return_attention_weights)
            if return_attention_weights and attn is not None:
                attn_list.append(attn.detach())

        global_rep = self._global_pool(x)
        out = {
            'avg_delay': self.delay_head(global_rep).squeeze(-1),
            'avg_loss': self.loss_head(global_rep).squeeze(-1),
            'avg_throughput': self.throughput_head(global_rep).squeeze(-1),
            'node_embeddings': x,
        }
        if flow_pairs is not None and flow_pairs.numel() > 0:
            src = x[flow_pairs[:, 0]]
            dst = x[flow_pairs[:, 1]]
            g = global_rep.expand(flow_pairs.size(0), -1)
            combined = torch.cat([src, dst, g], dim=-1)
            out['flow_delays'] = self.flow_head(combined).squeeze(-1)
        if return_attention_weights:
            out['attention_weights'] = attn_list
        return out

    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        mean_p = x.mean(dim=0, keepdim=True)
        max_p, _ = x.max(dim=0, keepdim=True)
        attn_out, _ = self.global_attention(
            mean_p.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_p = attn_out.squeeze(0)
        return torch.cat([mean_p, max_p, attn_p], dim=-1)


# ===========================================================================
# 3b. Spatio-Temporal RouteNet — full model with GRU-recurrent ST layers
# ===========================================================================
class SpatioTemporalRouteNet(nn.Module):
    """
    Spatio-temporal GNN for end-to-end delay prediction in DYNAMIC
    satellite topologies.
    """

    def __init__(self,
                 node_feat_dim: int = 7,
                 edge_feat_dim: int = 9,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_mixed_precision: bool = True):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_precision = use_mixed_precision

        # ---- encoders ----
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ---- spatio-temporal stack ----
        # Each ST layer takes encoded edges (hidden_dim) → so edge_dim_in
        # for each layer is hidden_dim.
        self.st_layers = nn.ModuleList([
            SpatioTemporalLayer(hidden_dim=hidden_dim,
                                num_heads=num_heads,
                                dropout=dropout,
                                edge_dim_in=hidden_dim)
            for _ in range(num_layers)
        ])

        # ---- global attention pool ----
        self.global_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # ---- prediction heads (3 = mean + max + attn pool) ----
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.loss_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.throughput_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        # Per-flow head: concat(src_emb, dst_emb, global_rep) → delay
        # global_rep is hidden_dim*3, so input is hidden_dim*5
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def init_hidden_states(self,
                           num_nodes: int,
                           device: Optional[torch.device] = None
                           ) -> List[torch.Tensor]:
        """Return a fresh list of zero hidden states, one per ST layer."""
        if device is None:
            device = next(self.parameters()).device
        return [torch.zeros(num_nodes, self.hidden_dim, device=device)
                for _ in range(self.num_layers)]

    # ------------------------------------------------------------------
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                flow_pairs: Optional[torch.Tensor] = None,
                hidden_states: Optional[List[torch.Tensor]] = None,
                return_attention_weights: bool = False
                ) -> Dict[str, Any]:
        """Single-snapshot forward pass.

        Args
        ----
        node_features  : [N, node_feat_dim]
        edge_index     : [2, E]
        edge_features  : [E_undirected, edge_feat_dim] or None
        flow_pairs     : [F, 2] int tensor of (src, dst) node indices
        hidden_states  : list of [N, hidden_dim] tensors, one per ST
                         layer, from the previous snapshot. None ⇒ zero.

        Returns dict with keys:
          'avg_delay', 'avg_loss', 'avg_throughput',
          'flow_delays' (if flow_pairs given),
          'node_embeddings',
          'hidden_states' (to feed back into the next snapshot).
        """
        if self.use_mixed_precision and AMP_AVAILABLE and node_features.is_cuda:
            with autocast():
                return self._forward_impl(node_features, edge_index,
                                          edge_features, flow_pairs,
                                          hidden_states,
                                          return_attention_weights)
        return self._forward_impl(node_features, edge_index, edge_features,
                                  flow_pairs, hidden_states,
                                  return_attention_weights)

    def _forward_impl(self, node_features, edge_index, edge_features,
                      flow_pairs, hidden_states, return_attention_weights):
        N = node_features.size(0)
        device = node_features.device

        # Encode
        x = self.node_encoder(node_features)
        e = None
        if edge_features is not None and edge_features.size(0) > 0:
            e = self.edge_encoder(edge_features)

        # Hidden-state bookkeeping: init if missing or shape-mismatched
        if hidden_states is None or len(hidden_states) != self.num_layers:
            hidden_states = self.init_hidden_states(N, device)

        # Thread through the ST stack
        new_hidden_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.st_layers):
            h_prev = hidden_states[i]
            if h_prev is None or h_prev.size(0) != N \
                    or h_prev.size(1) != self.hidden_dim:
                h_prev = torch.zeros(N, self.hidden_dim, device=device)
            # The layer's output IS the new hidden state for this layer
            x = layer(x, edge_index, e, h_prev)
            new_hidden_states.append(x)

        # Global pool (mean + max + attention pool)
        global_rep = self._global_pool(x)

        out: Dict[str, Any] = {
            'avg_delay': self.delay_head(global_rep).squeeze(-1),
            'avg_loss': self.loss_head(global_rep).squeeze(-1),
            'avg_throughput': self.throughput_head(global_rep).squeeze(-1),
            'node_embeddings': x,
            'hidden_states': new_hidden_states,
        }
        if flow_pairs is not None and flow_pairs.numel() > 0:
            src = x[flow_pairs[:, 0]]
            dst = x[flow_pairs[:, 1]]
            g = global_rep.expand(flow_pairs.size(0), -1)
            combined = torch.cat([src, dst, g], dim=-1)
            out['flow_delays'] = self.flow_head(combined).squeeze(-1)
        return out

    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        mean_p = x.mean(dim=0, keepdim=True)
        max_p, _ = x.max(dim=0, keepdim=True)
        attn_out, _ = self.global_attention(
            mean_p.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_p = attn_out.squeeze(0)
        return torch.cat([mean_p, max_p, attn_p], dim=-1)

    # ------------------------------------------------------------------
    def forward_sequence(self,
                         snapshots: List[Dict[str, torch.Tensor]],
                         flow_pairs: Optional[torch.Tensor] = None,
                         detach_between: bool = False
                         ) -> List[Dict[str, Any]]:
    
        hidden_states: Optional[List[torch.Tensor]] = None
        outputs: List[Dict[str, Any]] = []
        for snap in snapshots:
            out = self.forward(
                node_features=snap['node_features'],
                edge_index=snap['edge_index'],
                edge_features=snap.get('edge_features'),
                flow_pairs=flow_pairs,
                hidden_states=hidden_states,
            )
            hidden_states = out['hidden_states']
            if detach_between:
                hidden_states = [h.detach() for h in hidden_states]
            outputs.append(out)
        return outputs


# ===========================================================================
# 4. Simpler GNN for speed
# ===========================================================================
class SimpleGNN(nn.Module):
    """Lightweight message-passing GNN, much faster to train."""

    def __init__(self, node_dim: int = 7, edge_dim: int = 9,
                 hidden_dim: int = 64, output_dim: int = 1,
                 num_mp: int = 3, use_mixed_precision: bool = True):
        super().__init__()
        self.use_mixed_precision = use_mixed_precision
        self.hidden_dim = hidden_dim
        self.node_feat_dim = node_dim
        self.edge_feat_dim = edge_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message_passing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_mp)
        ])
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, node_features, edge_index, edge_features=None,
                flow_pairs=None, **kwargs):
        if self.use_mixed_precision and AMP_AVAILABLE and node_features.is_cuda:
            with autocast():
                return self._forward_impl(node_features, edge_index,
                                          edge_features, flow_pairs)
        return self._forward_impl(node_features, edge_index,
                                  edge_features, flow_pairs)

    def _forward_impl(self, node_features, edge_index, edge_features, flow_pairs):
        x = self.node_encoder(node_features)
        if edge_features is not None and edge_features.size(0) > 0:
            e = self.edge_encoder(edge_features)
            # Broadcast to both directions if needed
            if e.size(0) * 2 == edge_index.size(1):
                e_full = torch.cat([e, e], dim=0)
            else:
                e_full = e[:edge_index.size(1)]
        else:
            e_full = torch.zeros((edge_index.size(1), self.hidden_dim),
                                 device=x.device)

        for mp in self.message_passing:
            agg = torch.zeros_like(x)
            agg.index_add_(0, edge_index[1], e_full[:edge_index.size(1)])
            x = x + mp(torch.cat([x, agg], dim=-1))

        mean_p = x.mean(dim=0, keepdim=True)
        max_p, _ = x.max(dim=0, keepdim=True)
        min_p, _ = x.min(dim=0, keepdim=True)
        global_rep = torch.cat([mean_p, max_p, min_p], dim=-1)
        avg_delay = self.pooling(global_rep)

        out = {'avg_delay': avg_delay, 'node_embeddings': x}
        # Per-flow delay = average of source & destination embedding norms,
        # scaled by global avg delay → cheap heuristic but trainable
        if flow_pairs is not None and flow_pairs.numel() > 0:
            src_e = x[flow_pairs[:, 0]]
            dst_e = x[flow_pairs[:, 1]]
            sim = (src_e * dst_e).sum(dim=-1)
            flow_d = avg_delay.squeeze() * (1.0 + 0.1 * torch.tanh(sim))
            out['flow_delays'] = flow_d
        return out


# ===========================================================================
# 5. Predictor wrapper
# ===========================================================================
class GNNDelayPredictor:
    """High-level wrapper used by the rest of the code base."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 model_type: str = 'simple',
                 use_multi_gpu: bool = False,
                 use_amp: bool = True,
                 use_checkpointing: bool = False,
                 node_feat_dim: int = 7,
                 edge_feat_dim: int = 9,
                 hidden_dim: int = 64):
        # Device
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       if device == 'auto' else torch.device(device))
        self.use_amp = use_amp and AMP_AVAILABLE and self.device.type == 'cuda'
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.use_checkpointing = use_checkpointing
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        self.model_type = model_type
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim

        if model_type == 'full':
            self.model = RouteNetFermiWithMultiHeadAttention(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=128,
                num_heads=8,
                num_layers=3,
                use_mixed_precision=self.use_amp,
            ).to(self.device)
        elif model_type == 'spatiotemporal':
            # Use hidden_dim from caller (default 64) unless they want larger
            st_hidden = max(hidden_dim, 128) if hidden_dim < 128 else hidden_dim
            self.model = SpatioTemporalRouteNet(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=st_hidden,
                num_heads=8,
                num_layers=3,
                use_mixed_precision=self.use_amp,
            ).to(self.device)
            self.hidden_dim = st_hidden
        else:
            self.model = SimpleGNN(
                node_dim=node_feat_dim,
                edge_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                use_mixed_precision=self.use_amp,
            ).to(self.device)

        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.scaler = GradScaler() if self.use_amp else None
        self.is_trained = False

        # Per-node hidden state carried across snapshots for the
        # spatio-temporal model. None ⇒ fresh start on next predict_delay.
        self._st_hidden_states: Optional[List[torch.Tensor]] = None
        self._st_last_num_nodes: Optional[int] = None

        if model_path:
            self.load_model(model_path)

        # cuDNN tuning
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        logger.info(
            f"GNNDelayPredictor: device={self.device}, type={model_type}, "
            f"AMP={self.use_amp}, MGPU={self.use_multi_gpu}")

    # ---------------- prediction ----------------
    def reset_temporal_state(self):
      
        self._st_hidden_states = None
        self._st_last_num_nodes = None

    def predict_delay(self, network_state: Dict[str, Any],
                      flow_pairs: Optional[np.ndarray] = None
                      ) -> Dict[str, Any]:
     
        try:
            inputs = self._prepare_model_inputs(network_state, flow_pairs)
            mdl = self.model.module if isinstance(self.model, nn.DataParallel) \
                else self.model
            mdl.eval()

            is_st = isinstance(mdl, SpatioTemporalRouteNet)
            if is_st:
                # Auto-reset if node count changed (e.g. between episodes
                # that the caller forgot to reset_temporal_state on)
                N = inputs['node_features'].size(0)
                if self._st_last_num_nodes is not None \
                        and self._st_last_num_nodes != N:
                    logger.warning(
                        f"Node count changed {self._st_last_num_nodes} → {N}; "
                        "auto-resetting spatio-temporal hidden state.")
                    self._st_hidden_states = None
                inputs['hidden_states'] = self._st_hidden_states
                self._st_last_num_nodes = N

            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        out = mdl(**inputs)
                else:
                    out = mdl(**inputs)

            # If ST model, snapshot the new hidden state (detached, on
            # the model's device) for the NEXT call.
            if is_st and 'hidden_states' in out:
                self._st_hidden_states = [h.detach()
                                          for h in out['hidden_states']]

            result = {}
            for k, v in out.items():
                if k == 'hidden_states':
                    # Don't ship internal state to caller — they don't need it
                    continue
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        result[k] = float(v.item())
                    elif v.numel() == 1:
                        result[k] = float(v.item())
                    else:
                        result[k] = v.detach().cpu().numpy()
                else:
                    result[k] = v
            result['confidence'] = 0.9 if self.is_trained else 0.5
            return result
        except Exception as e:
            logger.debug(f"GNN prediction failed: {e}")
            return {'avg_delay': 0.0, 'confidence': 0.0}

    def _prepare_model_inputs(self, network_state: Dict[str, Any],
                              flow_pairs: Optional[np.ndarray] = None
                              ) -> Dict[str, torch.Tensor]:
        node_feats = network_state.get('node_features')
        edge_feats = network_state.get('edge_features')
        edge_index = network_state.get('edge_index')

        if node_feats is None or len(node_feats) == 0:
            node_feats = np.zeros((1, self.node_feat_dim), dtype=np.float32)

        # Pad / truncate to the expected feature dimensions to avoid crashes
        if node_feats.shape[1] != self.node_feat_dim:
            n = node_feats.shape[0]
            new = np.zeros((n, self.node_feat_dim), dtype=np.float32)
            c = min(node_feats.shape[1], self.node_feat_dim)
            new[:, :c] = node_feats[:, :c]
            node_feats = new

        if edge_feats is None:
            edge_feats = np.zeros((1, self.edge_feat_dim), dtype=np.float32)
        elif edge_feats.shape[1] != self.edge_feat_dim:
            n = edge_feats.shape[0]
            new = np.zeros((n, self.edge_feat_dim), dtype=np.float32)
            c = min(edge_feats.shape[1], self.edge_feat_dim)
            new[:, :c] = edge_feats[:, :c]
            edge_feats = new

        if edge_index is None or len(edge_index) == 0 \
                or (hasattr(edge_index, 'shape') and edge_index.shape[1] == 0):
            # Fall-back: self-loops only
            n = node_feats.shape[0]
            arr = np.arange(n)
            edge_index = np.stack([arr, arr])

        d = {
            'node_features': torch.as_tensor(node_feats, dtype=torch.float32,
                                             device=self.device),
            'edge_index': torch.as_tensor(edge_index, dtype=torch.long,
                                          device=self.device),
            'edge_features': torch.as_tensor(edge_feats, dtype=torch.float32,
                                             device=self.device),
        }
        if flow_pairs is not None and len(flow_pairs) > 0:
            d['flow_pairs'] = torch.as_tensor(flow_pairs, dtype=torch.long,
                                              device=self.device)
        return d

    # ---------------- training ----------------
    def train(self, train_data: Dict[str, Any],
              val_data: Optional[Dict[str, Any]] = None,
              epochs: int = 50, lr: float = 1e-3,
              batch_size: int = 32, gradient_accumulation_steps: int = 1):
       
        mdl = self.model.module if isinstance(self.model, nn.DataParallel) \
            else self.model
        mdl.train()
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(5, epochs // 4), T_mult=2)
        crit = nn.SmoothL1Loss()

        graphs = train_data.get('graphs', None)  # list of dicts
        if graphs is None:
            logger.warning("Training data must be a list of graph dicts; "
                           "got legacy format — skipping training.")
            return {}

        stats = {'train_loss': [], 'val_loss': [], 'lr': []}

        for ep in range(epochs):
            np.random.shuffle(graphs)
            ep_loss = 0.0
            opt.zero_grad()
            for i, g in enumerate(graphs):
                inp = self._prepare_model_inputs(g)
                target = torch.tensor(g['target_delay'], dtype=torch.float32,
                                      device=self.device).view(1)

                if self.use_amp:
                    with autocast():
                        out = mdl(**inp)
                        pred = out['avg_delay'].view(1)
                        loss = crit(pred, target) / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                        self.scaler.step(opt); self.scaler.update(); opt.zero_grad()
                else:
                    out = mdl(**inp)
                    pred = out['avg_delay'].view(1)
                    loss = crit(pred, target) / gradient_accumulation_steps
                    loss.backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                        opt.step(); opt.zero_grad()
                ep_loss += loss.item() * gradient_accumulation_steps

            sched.step()
            stats['train_loss'].append(ep_loss / max(len(graphs), 1))
            stats['lr'].append(sched.get_last_lr()[0])

            if val_data is not None and (ep + 1) % 10 == 0:
                val_loss = self._eval(val_data.get('graphs', []), crit)
                stats['val_loss'].append(val_loss)
            else:
                stats['val_loss'].append(0.0)

            if (ep + 1) % 10 == 0:
                logger.info(f"GNN epoch {ep+1}/{epochs}: "
                            f"train_loss={stats['train_loss'][-1]:.4f}, "
                            f"lr={stats['lr'][-1]:.2e}")

        self.is_trained = True
        logger.info("GNN training complete")
        return stats

    def _eval(self, graphs: List[Dict[str, Any]], crit) -> float:
        mdl = self.model.module if isinstance(self.model, nn.DataParallel) \
            else self.model
        mdl.eval()
        total = 0.0
        with torch.no_grad():
            for g in graphs:
                inp = self._prepare_model_inputs(g)
                target = torch.tensor(g['target_delay'], dtype=torch.float32,
                                      device=self.device).view(1)
                out = mdl(**inp)
                pred = out['avg_delay'].view(1)
                total += crit(pred, target).item()
        mdl.train()
        return total / max(len(graphs), 1)

    # ---------------- sequence training (BPTT) ----------------
    def train_sequence(self,
                       sequences_data: Dict[str, Any],
                       epochs: int = 50,
                       lr: float = 1e-3,
                       truncate_bptt: int = 0,
                       gradient_accumulation_steps: int = 1):
        """BPTT training on chronological sequences of topology snapshots.
        """
        mdl = self.model.module if isinstance(self.model, nn.DataParallel) \
            else self.model
        if not isinstance(mdl, SpatioTemporalRouteNet):
            logger.warning("train_sequence requires model_type='spatiotemporal'; "
                           "skipping.")
            return {}

        mdl.train()
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(5, epochs // 4), T_mult=2)
        crit = nn.SmoothL1Loss()

        sequences = sequences_data.get('sequences', None)
        if not sequences:
            logger.warning("train_sequence: empty 'sequences' — skipping.")
            return {}

        stats = {'train_loss': [], 'lr': []}

        for ep in range(epochs):
            np.random.shuffle(sequences)
            ep_loss = 0.0
            ep_steps = 0
            opt.zero_grad()

            for s_idx, seq in enumerate(sequences):
                if not seq:
                    continue

                # Snapshot 0 — initialise hidden state to zeros
                first = self._prepare_model_inputs(seq[0])
                N = first['node_features'].size(0)
                hidden_states = mdl.init_hidden_states(N, self.device)

                seq_loss = 0.0
                for t, snap in enumerate(seq):
                    inp = self._prepare_model_inputs(snap)
                    target = torch.tensor(snap['target_delay'],
                                          dtype=torch.float32,
                                          device=self.device).view(1)

                    inp['hidden_states'] = hidden_states
                    if self.use_amp:
                        with autocast():
                            out = mdl(**inp)
                            pred = out['avg_delay'].view(1)
                            step_loss = crit(pred, target)
                    else:
                        out = mdl(**inp)
                        pred = out['avg_delay'].view(1)
                        step_loss = crit(pred, target)

                    seq_loss = seq_loss + step_loss
                    hidden_states = out['hidden_states']

                    # Truncated BPTT: detach periodically to bound memory
                    if truncate_bptt > 0 and (t + 1) % truncate_bptt == 0:
                        hidden_states = [h.detach() for h in hidden_states]

                # Backprop once per sequence
                seq_loss = seq_loss / max(len(seq), 1) / gradient_accumulation_steps
                if self.use_amp:
                    self.scaler.scale(seq_loss).backward()
                else:
                    seq_loss.backward()

                if (s_idx + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                        self.scaler.step(opt); self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                        opt.step()
                    opt.zero_grad()

                ep_loss += float(seq_loss.item()) * gradient_accumulation_steps
                ep_steps += 1

            sched.step()
            stats['train_loss'].append(ep_loss / max(ep_steps, 1))
            stats['lr'].append(sched.get_last_lr()[0])

            if (ep + 1) % 10 == 0:
                logger.info(f"ST-GNN epoch {ep+1}/{epochs}: "
                            f"train_loss={stats['train_loss'][-1]:.4f}, "
                            f"lr={stats['lr'][-1]:.2e}")

        self.is_trained = True
        logger.info("Spatio-temporal GNN training complete")
        return stats

    # ---------------- IO ----------------
    def save_model(self, path: str):
        sd = self.model.module.state_dict() \
            if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model_state_dict': sd,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'use_multi_gpu': self.use_multi_gpu,
            'use_amp': self.use_amp,
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'hidden_dim': self.hidden_dim,
        }, path)
        logger.info(f"GNN model saved → {path}")

    def load_model(self, path: str):
        try:
            ck = torch.load(path, map_location=self.device)
            self.node_feat_dim = ck.get('node_feat_dim', self.node_feat_dim)
            self.edge_feat_dim = ck.get('edge_feat_dim', self.edge_feat_dim)
            self.hidden_dim = ck.get('hidden_dim', self.hidden_dim)
            self.model_type = ck.get('model_type', self.model_type)

            if self.model_type == 'full':
                self.model = RouteNetFermiWithMultiHeadAttention(
                    node_feat_dim=self.node_feat_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    hidden_dim=128, use_mixed_precision=self.use_amp,
                ).to(self.device)
            elif self.model_type == 'spatiotemporal':
                st_hidden = max(self.hidden_dim, 128) \
                    if self.hidden_dim < 128 else self.hidden_dim
                self.model = SpatioTemporalRouteNet(
                    node_feat_dim=self.node_feat_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    hidden_dim=st_hidden,
                    num_heads=8,
                    num_layers=3,
                    use_mixed_precision=self.use_amp,
                ).to(self.device)
                self.hidden_dim = st_hidden
            else:
                self.model = SimpleGNN(
                    node_dim=self.node_feat_dim,
                    edge_dim=self.edge_feat_dim,
                    hidden_dim=self.hidden_dim,
                    use_mixed_precision=self.use_amp,
                ).to(self.device)
            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            sd = ck['model_state_dict']
            # Adjust DataParallel prefix
            if isinstance(self.model, nn.DataParallel) and not any(
                    k.startswith('module.') for k in sd):
                sd = {'module.' + k: v for k, v in sd.items()}
            elif not isinstance(self.model, nn.DataParallel) and any(
                    k.startswith('module.') for k in sd):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
            self.is_trained = ck.get('is_trained', False)
            logger.info(f"GNN model loaded ← {path}")
        except Exception as e:
            logger.error(f"Failed to load GNN model from {path}: {e}")
            self.is_trained = False


# ===========================================================================
# 6. Backward-compatible aliases
# ===========================================================================
class RouteNet_Fermi_with_Attention(RouteNetFermiWithMultiHeadAttention):
    pass


class RouteNet_Fermi_with_MultiHeadAttention(RouteNetFermiWithMultiHeadAttention):
    pass


def create_attention_model(traffic_model: str = 'all_multiplexed'):
    return RouteNetFermiWithMultiHeadAttention()


# ===========================================================================
# 7. Self-test
# ===========================================================================
def test_gnn_model():
    print("=" * 60)
    print("Testing GNNDelayPredictor (simple)…")
    print("=" * 60)
    pred = GNNDelayPredictor(model_type='simple', use_amp=False)
    state = {
        'node_features': np.random.randn(20, 7).astype(np.float32),
        'edge_features': np.random.randn(30, 9).astype(np.float32),
        'edge_index': np.stack([np.random.randint(0, 20, 60),
                                np.random.randint(0, 20, 60)]),
    }
    out = pred.predict_delay(state, flow_pairs=np.array([[0, 1], [2, 3]]))
    print(f"  prediction keys : {list(out.keys())}")
    print(f"  avg_delay       : {out.get('avg_delay')}")
    print("  OK.")

    print("\n" + "=" * 60)
    print("Testing GNNDelayPredictor (spatiotemporal)…")
    print("=" * 60)
    st_pred = GNNDelayPredictor(model_type='spatiotemporal',
                                use_amp=False,
                                hidden_dim=64)
    N = 29   # fixed satellite count
    flow_pairs = np.array([[0, 5], [3, 10], [7, 15]])

    # Simulate a chronological sequence of 4 topology snapshots — each
    # with slightly different edges (ISL handovers) but same node count.
    rng = np.random.RandomState(0)
    prev_delay = None
    for t in range(4):
        # Random topology — different edges each timestep
        E = 40
        edge_index = np.stack([rng.randint(0, N, E),
                               rng.randint(0, N, E)])
        snapshot = {
            'node_features': rng.randn(N, 7).astype(np.float32),
            'edge_features': rng.randn(E // 2, 9).astype(np.float32),
            'edge_index': edge_index,
        }
        out = st_pred.predict_delay(snapshot, flow_pairs=flow_pairs)
        # Confirm hidden state is being carried (not None after first call)
        h_alive = st_pred._st_hidden_states is not None
        h_layers = len(st_pred._st_hidden_states) if h_alive else 0
        print(f"  t={t}: avg_delay={out['avg_delay']:.4f}  "
              f"flow_delays_shape={getattr(out.get('flow_delays'), 'shape', None)}  "
              f"hidden_alive={h_alive} ({h_layers} layers)")
        prev_delay = out['avg_delay']

    # Reset and verify the hidden state is cleared
    st_pred.reset_temporal_state()
    assert st_pred._st_hidden_states is None, "reset_temporal_state failed"
    print("  reset_temporal_state OK.")

    # Sanity-check: predicting AFTER reset should not crash
    out2 = st_pred.predict_delay(snapshot, flow_pairs=flow_pairs)
    print(f"  post-reset avg_delay = {out2['avg_delay']:.4f}")
    print("  OK.")
    print("\nAll tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gnn_model()
