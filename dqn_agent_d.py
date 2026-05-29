"""
dqn_agent_d.py - Delay-Aware Dueling DQN Agent
"""

import logging
import math
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DQNConfig
from dqn_components import (QNetwork, DuelingQNetwork, ReplayBuffer,
                            NoisyQNetwork)
from gnn_model import GNNDelayPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe-batchnorm
# ---------------------------------------------------------------------------
class SafeBatchNorm1d(nn.Module):
    """BatchNorm1d that survives batch-size 1 during training."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps,
                                 momentum=momentum, affine=affine)

    def forward(self, x):
        if x.size(0) == 1 and self.training:
            self.bn.eval()
            out = self.bn(x)
            self.bn.train()
            return out
        return self.bn(x)


# ---------------------------------------------------------------------------
# Delay-aware replay buffer
# ---------------------------------------------------------------------------
class DelayAwareReplayBuffer:
    """PER buffer that additionally up-weights high-delay transitions."""

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.buffer: List[Optional[Tuple]] = []
        self.position = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities: List[float] = []
        self.delay_weights: List[float] = []
        self.max_priority = 1.0
        self.delay_threshold = 200.0   # ms

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done, delay: float = 0.0,
            priority: Optional[float] = None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(0.0)
            self.delay_weights.append(1.0)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        # FIX: new transitions start with max priority (or caller-supplied
        # priority if provided, e.g. from train_dqn_s.py's priority hint)
        self.priorities[self.position] = (
            float(priority) if priority is not None else self.max_priority
        )

        if delay > self.delay_threshold:
            self.delay_weights[self.position] = 1.0 + \
                (delay - self.delay_threshold) / 200.0   # bounded growth
        else:
            self.delay_weights[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None

        prios = np.array(self.priorities[:len(self.buffer)], dtype=np.float64)
        probs = prios ** self.alpha
        s = probs.sum()
        probs = np.ones_like(probs) / len(probs) if s == 0 else probs / s

        idx = np.random.choice(len(self.buffer), batch_size, p=probs,
                               replace=False)
        weights = (len(self.buffer) * probs[idx]) ** (-self.beta)
        weights = weights / max(weights.max(), 1e-9)
        self.beta = min(1.0, self.beta + self.beta_increment)

        dw = np.array([self.delay_weights[i] for i in idx])
        weights = (weights * dw).astype(np.float32)

        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                idx, weights)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            p = float((abs(td) + 1e-6) ** self.alpha)
            self.priorities[i] = p
            self.max_priority = max(self.max_priority, p)


# ---------------------------------------------------------------------------
# Delay-aware Q network
# ---------------------------------------------------------------------------
class DelayAwareQNetwork(nn.Module):
    """Dueling Q-network with an auxiliary delay-prediction head."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 use_safe_batchnorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        in_dim = state_dim
        for hd in hidden_dims:
            layers += [
                nn.Linear(in_dim, hd),
                SafeBatchNorm1d(hd) if use_safe_batchnorm else nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ]
            in_dim = hd
        self.feature_extractor = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        # Aux head — use Softplus so the gradient is non-trivial near 0
        self.delay_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_delay: bool = False):
        f = self.feature_extractor(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        if return_delay:
            return q, self.delay_predictor(f)
        return q


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """Dueling DQN + Delay-aware learning + GNN-based path delay queries."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: DQNConfig,
                 gnn_predictor: GNNDelayPredictor,
                 use_dueling: bool = True,
                 use_noisy: bool = False,
                 use_delay_aware: bool = True,
                 use_safe_batchnorm: bool = True,
                 device: Optional[str] = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.gnn_predictor = gnn_predictor
        self.use_delay_aware = use_delay_aware
        self.use_safe_batchnorm = use_safe_batchnorm

        self.device = torch.device(device) if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- networks -----------------------------------------------------
        if use_delay_aware:
            self.q_network = DelayAwareQNetwork(
                state_dim, action_dim,
                hidden_dims=config.HIDDEN_DIMS,
                use_safe_batchnorm=use_safe_batchnorm).to(self.device)
            self.target_network = DelayAwareQNetwork(
                state_dim, action_dim,
                hidden_dims=config.HIDDEN_DIMS,
                use_safe_batchnorm=use_safe_batchnorm).to(self.device)
        elif use_noisy:
            self.q_network = NoisyQNetwork(state_dim, action_dim).to(self.device)
            self.target_network = NoisyQNetwork(state_dim, action_dim).to(self.device)
        elif use_dueling:
            self.q_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_network = QNetwork(state_dim, action_dim,
                                      hidden_dims=config.HIDDEN_DIMS).to(self.device)
            self.target_network = QNetwork(state_dim, action_dim,
                                           hidden_dims=config.HIDDEN_DIMS).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # AdamW: weight_decay is the only L2 (no double regularisation)
        self.optimizer = optim.AdamW(self.q_network.parameters(),
                                     lr=config.LR,
                                     weight_decay=config.WEIGHT_DECAY)
        # ReduceLROnPlateau watches loss, but loss going down doesn't mean
        # policy improving — Q-values just fit buffered targets. Kept for
        # divergence protection but with high patience so it rarely fires.
        # Real LR decay is the per-episode cosine schedule below.
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50)
        # Per-episode cosine annealing — call set_lr_schedule(num_episodes)
        # at start of training. LR goes LR → LR/10 across all episodes.
        self._lr_initial = float(config.LR)
        self._lr_floor = max(float(config.LR_MIN), self._lr_initial / 10.0)
        self._lr_total_episodes = 0

        # --- buffer & training state -------------------------------------
        self.replay_buffer = DelayAwareReplayBuffer(
            config.BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            beta=config.PER_BETA,
            beta_increment=config.PER_BETA_INCREMENT,
        )
        self.epsilon = config.EPS_START
        self.epsilon_min = config.EPS_END
        self.epsilon_decay = config.EPS_DECAY
        self.learn_step_counter = 0

        # Tracking
        self.episode_rewards: List[float] = []
        self.loss_history: List[float] = []
        self.performance_window: Deque[float] = deque(maxlen=config.PERFORMANCE_WINDOW)
        self.delay_window: Deque[float] = deque(maxlen=config.PERFORMANCE_WINDOW)
        self.avg_delay_history: List[float] = []
        self.qos_violation_history: List[float] = []
        self.success_rate_history: List[float] = []

        self.exploration_stats = {
            'random_actions': 0,
            'greedy_actions': 0,
            'total_actions': 0,
        }

        # Loss weights
        self.delay_penalty_weight = 0.1
        self.delay_prediction_weight = 0.05
        self.max_allowed_delay = 200.0  # ms

        logger.info(f"DQNAgent on {self.device}  "
                    f"state_dim={state_dim} action_dim={action_dim} "
                    f"delay_aware={use_delay_aware}")

    # ----------------------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------------------
    def select_action(self, state: np.ndarray, training: bool = True,
                      epsilon: Optional[float] = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon if training else 0.05
        self.exploration_stats['total_actions'] += 1

        if training and random.random() < epsilon:
            self.exploration_stats['random_actions'] += 1
            return random.randint(0, self.action_dim - 1)

        self.exploration_stats['greedy_actions'] += 1
        st = torch.as_tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            if self.use_delay_aware:
                q, _ = self.q_network(st, return_delay=True)
            else:
                q = self.q_network(st)
        self.q_network.train()
        return int(torch.argmax(q, dim=1).item())

    # ----------------------------------------------------------------------
    # GNN-assisted path delay query (used by the trainer)
    # ----------------------------------------------------------------------
    def predict_path_delay(self, network_state: Dict[str, Any],
                           src_idx: int, dst_idx: int) -> float:
        """Query the GNN model for an estimated path delay in **ms**.

        Returns 0.0 if GNN is untrained or prediction fails so callers
        can fall back to heuristics gracefully.
        """
        try:
            out = self.gnn_predictor.predict_delay(
                network_state,
                flow_pairs=np.array([[src_idx, dst_idx]], dtype=np.int64),
            )
            if 'flow_delays' in out:
                fd = out['flow_delays']
                if isinstance(fd, np.ndarray) and fd.size > 0:
                    return float(fd[0])
                return float(fd)
            return float(out.get('avg_delay', 0.0))
        except Exception as e:
            logger.debug(f"predict_path_delay failed: {e}")
            return 0.0

    # ----------------------------------------------------------------------
    # Buffer interaction
    # ----------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done,
                         delay: float = 0.0, priority: Optional[float] = None):
        """Store a transition. `priority` is accepted for compatibility
        """
        # If the buffer accepts a priority argument we forward it; otherwise
        # we just store the transition normally.
        try:
            self.replay_buffer.add(state, action, reward, next_state,
                                   float(done), delay, priority=priority)
        except TypeError:
            self.replay_buffer.add(state, action, reward, next_state,
                                   float(done), delay)

    # ----------------------------------------------------------------------
    # Learning step
    # ----------------------------------------------------------------------
    def learn(self, batch_size: Optional[int] = None) -> float:
        bs = batch_size or self.config.BATCH_SIZE
        if len(self.replay_buffer) < max(bs, self.config.LEARNING_STARTS):
            return 0.0
        sample = self.replay_buffer.sample(bs)
        if sample is None:
            return 0.0
        states, actions, rewards, next_states, dones, indices, weights = sample

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        self.q_network.train()

        # Current Q values
        if self.use_delay_aware:
            current_q, delay_pred = self.q_network(s_t, return_delay=True)
            current_q = current_q.gather(1, a_t.unsqueeze(1)).squeeze(1)
        else:
            current_q = self.q_network(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            delay_pred = None

        # Target Q values (Double DQN)
        with torch.no_grad():
            if self.config.USE_DOUBLE_DQN:
                if self.use_delay_aware:
                    online_next, _ = self.q_network(ns_t, return_delay=True)
                    target_next, _ = self.target_network(ns_t, return_delay=True)
                else:
                    online_next = self.q_network(ns_t)
                    target_next = self.target_network(ns_t)
                next_a = online_next.argmax(dim=1, keepdim=True)
                next_q = target_next.gather(1, next_a).squeeze(1)
            else:
                if self.use_delay_aware:
                    target_next, _ = self.target_network(ns_t, return_delay=True)
                else:
                    target_next = self.target_network(ns_t)
                next_q = target_next.max(dim=1)[0]
            target_q = r_t + (1.0 - d_t) * self.config.GAMMA * next_q

        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # PER-weighted Huber loss (more robust than MSE)
        elementwise = F.smooth_l1_loss(current_q, target_q, reduction='none')
        main_loss = (w_t * elementwise).mean()

        # Auxiliary delay prediction
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.use_delay_aware and delay_pred is not None:
            # Target is normalised "allowed" delay
            tgt = torch.full_like(delay_pred,
                                  self.max_allowed_delay / 1000.0)
            aux_loss = F.smooth_l1_loss(delay_pred, tgt)

        total_loss = main_loss + self.delay_prediction_weight * aux_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                       self.config.GRADIENT_CLIP)
        self.optimizer.step()

        # Soft target update
        self._soft_update_target_network()

        self.learn_step_counter += 1
        self.loss_history.append(total_loss.item())
        if self.learn_step_counter % 200 == 0:
            self.lr_scheduler.step(total_loss.item())
        return total_loss.item()

    def _soft_update_target_network(self):
        tau = self.config.TAU
        with torch.no_grad():
            for tp, lp in zip(self.target_network.parameters(),
                              self.q_network.parameters()):
                tp.data.mul_(1.0 - tau).add_(lp.data, alpha=tau)

    # ----------------------------------------------------------------------
    # PER-compatible learn step used by train_dqn_s.py
    # ----------------------------------------------------------------------
    def learn_with_priorities(self, states, actions, rewards, next_states, dones, weights):
        """Learn from an externally-sampled batch with IS weights.
        """
        try:
            # Coerce to the right device/dtype
            s_t = states.to(self.device, dtype=torch.float32)
            a_t = actions.to(self.device, dtype=torch.long).view(-1)
            r_t = rewards.to(self.device, dtype=torch.float32).view(-1)
            ns_t = next_states.to(self.device, dtype=torch.float32)
            d_t = dones.to(self.device, dtype=torch.float32).view(-1)
            w_t = weights.to(self.device, dtype=torch.float32).view(-1)

            self.q_network.train()

            if self.use_delay_aware:
                current_q_all, delay_pred = self.q_network(s_t, return_delay=True)
            else:
                current_q_all = self.q_network(s_t)
                delay_pred = None
            current_q = current_q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.config.USE_DOUBLE_DQN:
                    if self.use_delay_aware:
                        online_next, _ = self.q_network(ns_t, return_delay=True)
                        target_next, _ = self.target_network(ns_t, return_delay=True)
                    else:
                        online_next = self.q_network(ns_t)
                        target_next = self.target_network(ns_t)
                    next_a = online_next.argmax(dim=1, keepdim=True)
                    next_q = target_next.gather(1, next_a).squeeze(1)
                else:
                    if self.use_delay_aware:
                        target_next, _ = self.target_network(ns_t, return_delay=True)
                    else:
                        target_next = self.target_network(ns_t)
                    next_q = target_next.max(dim=1)[0]
                target_q = r_t + (1.0 - d_t) * self.config.GAMMA * next_q

            td_errors = (current_q - target_q).detach()
            elementwise = F.smooth_l1_loss(current_q, target_q, reduction='none')
            main_loss = (w_t * elementwise).mean()

            aux_loss = torch.tensor(0.0, device=self.device)
            if self.use_delay_aware and delay_pred is not None:
                tgt = torch.full_like(delay_pred, self.max_allowed_delay / 1000.0)
                aux_loss = F.smooth_l1_loss(delay_pred, tgt)
            total_loss = main_loss + self.delay_prediction_weight * aux_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),
                                           self.config.GRADIENT_CLIP)
            self.optimizer.step()
            self._soft_update_target_network()

            self.learn_step_counter += 1
            self.loss_history.append(total_loss.item())
            if self.learn_step_counter % 200 == 0:
                self.lr_scheduler.step(total_loss.item())

            return np.abs(td_errors.cpu().numpy()) + 1e-6
        except Exception as e:
            logger.warning(f"learn_with_priorities failed: {e}")
            return None

    # ----------------------------------------------------------------------
    # Exploration control
    # ----------------------------------------------------------------------
    def update_epsilon(self):
        """Default multiplicative decay (used when no schedule is set).
        """
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def set_epsilon_schedule(self, total_episodes: int,
                             decay_fraction: float = 0.7):
        """Configure linear epsilon decay tied to a known training length.
        """
        self._sched_total = max(int(total_episodes * decay_fraction), 1)
        self._sched_start = float(self.config.EPS_START)
        self._sched_end = float(self.config.EPS_END)
        self.epsilon = self._sched_start
        logger.info(
            f"Epsilon schedule: {self._sched_start:.3f} -> {self._sched_end:.3f} "
            f"over {self._sched_total} episodes"
        )
        # Also configure the cosine LR schedule for the same horizon
        self._lr_total_episodes = max(int(total_episodes), 1)
        logger.info(
            f"LR schedule (cosine): {self._lr_initial:.2e} -> "
            f"{self._lr_floor:.2e} over {self._lr_total_episodes} episodes"
        )

    def set_lr_for_episode(self, episode_idx: int):
        """Per-episode cosine LR annealing.
        """
        if self._lr_total_episodes <= 0:
            return
        import math
        frac = min(1.0, max(0.0, episode_idx / self._lr_total_episodes))
        lr = (self._lr_floor
              + 0.5 * (self._lr_initial - self._lr_floor)
              * (1.0 + math.cos(math.pi * frac)))
        for g in self.optimizer.param_groups:
            g['lr'] = float(lr)

    def decay_epsilon_for_episode(self, episode_idx: int):
        """Set epsilon based on episode index (0-indexed).
        """
        if not hasattr(self, '_sched_total'):
            self.update_epsilon()
            return
        frac = min(1.0, episode_idx / self._sched_total)
        self.epsilon = self._sched_start + (self._sched_end - self._sched_start) * frac
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_exploration_rate(self) -> float:
        tot = max(self.exploration_stats['total_actions'], 1)
        return self.exploration_stats['random_actions'] / tot

    def reset_exploration_stats(self):
        self.exploration_stats = {'random_actions': 0,
                                  'greedy_actions': 0,
                                  'total_actions': 0}

    # ----------------------------------------------------------------------
    # Performance tracking
    # ----------------------------------------------------------------------
    def log_performance(self, episode_reward: float, avg_delay: float,
                        qos_violation_rate: float):
        self.performance_window.append(episode_reward)
        self.delay_window.append(avg_delay)
        self.avg_delay_history.append(avg_delay)
        self.qos_violation_history.append(qos_violation_rate)
        self.success_rate_history.append(1.0 - qos_violation_rate)
        self.episode_rewards.append(episode_reward)

    # ----------------------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------------------
    def save_model(self, path: str):
        ck = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'loss_history': self.loss_history[-500:],
            'exploration_stats': self.exploration_stats,
            'performance_window': list(self.performance_window),
            'delay_window': list(self.delay_window),
            'delay_penalty_weight': self.delay_penalty_weight,
            'delay_prediction_weight': self.delay_prediction_weight,
            'avg_delay_history': self.avg_delay_history[-500:],
            'qos_violation_history': self.qos_violation_history[-500:],
            'success_rate_history': self.success_rate_history[-500:],
        }
        torch.save(ck, path)
        logger.info(f"DQN model saved → {path}")

    def load_model(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ck['q_network_state_dict'])
        self.target_network.load_state_dict(ck['target_network_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.epsilon = ck.get('epsilon', self.epsilon)
        self.learn_step_counter = ck.get('learn_step_counter', 0)
        self.loss_history = ck.get('loss_history', [])
        self.exploration_stats = ck.get('exploration_stats',
                                        self.exploration_stats)
        self.performance_window = deque(ck.get('performance_window', []),
                                        maxlen=self.config.PERFORMANCE_WINDOW)
        self.delay_window = deque(ck.get('delay_window', []),
                                  maxlen=self.config.PERFORMANCE_WINDOW)
        self.delay_penalty_weight = ck.get('delay_penalty_weight',
                                           self.delay_penalty_weight)
        self.delay_prediction_weight = ck.get('delay_prediction_weight',
                                              self.delay_prediction_weight)
        self.avg_delay_history = ck.get('avg_delay_history', [])
        self.qos_violation_history = ck.get('qos_violation_history', [])
        self.success_rate_history = ck.get('success_rate_history', [])
        logger.info(f"DQN model loaded ← {path}")

    def get_training_stats(self) -> Dict[str, Any]:
        stats = {
            'episodes_completed': len(self.episode_rewards),
            'current_epsilon': self.epsilon,
            'learning_steps': self.learn_step_counter,
            'replay_buffer_size': len(self.replay_buffer),
            'exploration_rate': self.get_exploration_rate(),
            'delay_penalty_weight': self.delay_penalty_weight,
            'delay_prediction_weight': self.delay_prediction_weight,
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'gamma': self.config.GAMMA,
        }
        if self.episode_rewards:
            recent = self.episode_rewards[-10:]
            stats.update({
                'avg_reward_last_10': float(np.mean(recent)),
                'best_reward': float(np.max(self.episode_rewards)),
                'recent_reward_std': float(np.std(recent)) if len(recent) > 1 else 0.0,
            })
        if self.avg_delay_history:
            stats.update({
                'avg_delay_last_10': float(np.mean(self.avg_delay_history[-10:])),
                'best_delay': float(np.min(self.avg_delay_history)),
            })
        if self.loss_history:
            stats['avg_loss_last_100'] = float(np.mean(self.loss_history[-100:]))
        return stats
