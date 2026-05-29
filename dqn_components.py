"""
dqn_components.py - DQN Neural Network Components (PyTorch)
"""

import math
from collections import deque
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Standard Q network
# ============================================================================
class QNetwork(nn.Module):
    """Plain MLP Q network."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        in_dim = state_dim
        for hd in hidden_dims:
            layers += [nn.Linear(in_dim, hd),
                       nn.ReLU(inplace=True),
                       nn.LayerNorm(hd),
                       nn.Dropout(0.1)]
            in_dim = hd
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        logger.info(f"QNetwork: state={state_dim}, action={action_dim}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.forward(t).squeeze().cpu().numpy()


# ============================================================================
# Dueling Q network
# ============================================================================
class DuelingQNetwork(nn.Module):
    """Dueling architecture: V(s) + (A(s,a) - mean A(s,*))."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.apply(self._init_weights)
        logger.info(f"DuelingQNetwork: state={state_dim}, action={action_dim}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_layer(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# ============================================================================
# Prioritized replay buffer
# ============================================================================
class ReplayBuffer:
    """Prioritized experience replay with proportional sampling.
    """

    def __init__(self, buffer_size: int, alpha: float = 0.6, beta: float = 0.4,
                 recency_weight: float = 0.5):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        # Per-transition insertion timestamp (monotonic counter) for the
        # recency multiplier below. Same maxlen as the buffer so they
        # evict in lockstep.
        self.timestamps = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        # recency_weight controls how aggressively old transitions are
        # discounted. 0 = no recency bias (pure PER). 0.5 = oldest
        # transition in the buffer gets ~60% of its raw priority.
        # 1.0 = oldest gets ~37%. Tune up if buffer staleness persists.
        self.recency_weight = recency_weight
        self.step_counter = 0

    def add(self, state, action, reward, next_state, done, td_error=None):
        """Add a transition. New transitions get max_priority so they
        will be sampled at least once before being forgotten."""
        self.buffer.append((state, action, reward, next_state, done))
        self.timestamps.append(self.step_counter)
        self.step_counter += 1
        if td_error is None:
            self.priorities.append(self.max_priority)
        else:
            p = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities.append(p)
            self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size: int):
        n = len(self.buffer)
        if n == 0:
            return None
        bs = min(batch_size, n)

        prios = np.asarray(self.priorities, dtype=np.float64)
        # Recency boost: experiences from the recent tail of the buffer
        # get a higher effective probability. age_fraction is in [0,1]:
        # 0 = newest, 1 = oldest still in the buffer.
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        newest = timestamps.max()
        # Guard against the edge case of a single-element buffer
        span = max(newest - timestamps.min(), 1.0)
        age_fraction = (newest - timestamps) / span
        recency = np.exp(-self.recency_weight * age_fraction)
        effective_prios = prios * recency

        probs = effective_prios / max(effective_prios.sum(), 1e-9)
        idx = np.random.choice(n, size=bs, p=probs, replace=False)

        weights = (n * probs[idx]) ** (-self.beta)
        weights = weights / max(weights.max(), 1e-9)

        samples = [self.buffer[i] for i in idx]
        s, a, r, ns, d = zip(*samples)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32),
                idx,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            if i < len(self.priorities):
                p = (abs(float(td)) + 1e-6) ** self.alpha
                self.priorities[i] = p
                self.max_priority = max(self.max_priority, p)

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()
        self.timestamps.clear()
        self.max_priority = 1.0
        self.step_counter = 0

    def __len__(self):
        return len(self.buffer)

    def get_stats(self):
        return {
            'size': len(self.buffer),
            'capacity': self.buffer_size,
            'fill_rate': len(self.buffer) / self.buffer_size,
            'max_priority': self.max_priority,
            'avg_priority': float(np.mean(self.priorities)) if self.priorities else 0.0,
        }


# ============================================================================
# Noisy-net layer
# ============================================================================
class NoisyLinear(nn.Module):
    """Factorised noisy linear layer (Fortunato et al. 2018)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon',
                             torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        ein = self._scale_noise(self.in_features)
        eout = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eout.ger(ein))
        self.bias_epsilon.copy_(eout)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class NoisyQNetwork(nn.Module):
    """Q-network using NoisyLinear for exploration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            NoisyLinear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            NoisyLinear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.forward(t).squeeze().cpu().numpy()


# ============================================================================
# CNN feature extractor + Hybrid net  (kept for compatibility)
# ============================================================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 1, output_dim: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )
        self._output_dim = output_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc_layers(x.flatten(1))

    def get_output_dim(self) -> int:
        return self._output_dim


class HybridQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 traffic_matrix_shape: Tuple[int, int] = (50, 50),
                 hidden_dim: int = 256):
        super().__init__()
        self.cnn_extractor = CNNFeatureExtractor(1, 64)
        total_dim = state_dim + self.cnn_extractor.get_output_dim()
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor, traffic_matrix: torch.Tensor):
        cnn_feat = self.cnn_extractor(traffic_matrix)
        return self.mlp(torch.cat([state, cnn_feat], dim=1))
