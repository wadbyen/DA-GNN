"""
train_dqn_d.py - Trainer for the Delay-Aware Dueling DQN
"""

import argparse
import gc
import json
import logging
import os
import random
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # safe in headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (SatelliteConfig, TrafficConfig, DQNConfig,
                    SimulationConfig)
from environment_n import SatelliteNetworkEnvironment
from gnn_model import GNNDelayPredictor
from dqn_agent_d import DQNAgent

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State normalizer
# ---------------------------------------------------------------------------
class StateNormalizer:
    """Online z-score normaliser using Welford's algorithm."""

    def __init__(self, state_dim: int, alpha: float = 0.01,
                 clip_range: float = 5.0):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.var = np.ones(state_dim, dtype=np.float64)
        self.count = 1e-4
        self.clip_range = clip_range

    def update(self, state: np.ndarray):
        x = state.astype(np.float64)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = 0.99 * self.var + 0.01 * (delta * delta2)
        self.var = np.maximum(self.var, 1e-8)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var)
        z = (state.astype(np.float64) - self.mean) / std
        return np.clip(z, -self.clip_range, self.clip_range).astype(np.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DQNTrainer:
    """Train a DQN agent for delay-aware satellite routing."""

    def __init__(self,
                 sat_config: Optional[SatelliteConfig] = None,
                 traffic_config: Optional[TrafficConfig] = None,
                 dqn_config: Optional[DQNConfig] = None,
                 sim_config: Optional[SimulationConfig] = None,
                 use_multi_gpu: bool = False,
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 4,
                 delay_focus: bool = True,
                 output_dir: Optional[str] = None,
                 gnn_model_type: str = 'simple',
                 gnn_hidden_dim: int = 64,
                 fixed_lr: Optional[float] = None):

        self.sat_config = sat_config or SatelliteConfig()
        self.traffic_config = traffic_config or TrafficConfig()
        self.dqn_config = dqn_config or DQNConfig()
        self.sim_config = sim_config or SimulationConfig()

        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.use_amp = use_amp and AMP_AVAILABLE
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.delay_focus = delay_focus

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._enable_torch_optimizations()

        logger.info("Initializing environment...")
        self.env = SatelliteNetworkEnvironment(
            self.sat_config, self.traffic_config, self.sim_config)

        logger.info(f"Initializing GNN predictor (model_type={gnn_model_type})...")
        self.gnn_model_type = gnn_model_type
        self.gnn_predictor = GNNDelayPredictor(
            device=str(self.device),
            model_type=gnn_model_type,
            use_multi_gpu=self.use_multi_gpu,
            use_amp=self.use_amp,
            node_feat_dim=self.env.NODE_FEAT_DIM,
            edge_feat_dim=self.env.EDGE_FEAT_DIM,
            hidden_dim=gnn_hidden_dim,
        )

        # ** keep dims in lock-step with config (was hard-coded 28 / 3) **
        self.state_dim = self.dqn_config.STATE_DIM
        self.action_dim = self.dqn_config.ACTION_DIM

        logger.info("Initializing DQN agent...")
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.dqn_config,
            gnn_predictor=self.gnn_predictor,
            use_dueling=True,
            use_noisy=False,
            use_delay_aware=delay_focus,
            use_safe_batchnorm=True,
            device=str(self.device),
        )

        self.state_normalizer = StateNormalizer(self.state_dim)

        # Fixed-LR ablation: pin eta to a constant if requested. Applied
        # here so it survives; re-asserted in train() after the schedule
        # horizon is configured.
        self.fixed_lr = fixed_lr
        if fixed_lr is not None and hasattr(self.agent, 'set_fixed_lr'):
            self.agent.set_fixed_lr(fixed_lr)

        # node-name → node-index map for GNN queries
        self._node_idx_map = self.env._node_idx_map

        # Training metrics
        self.training_metrics: Dict[str, List[float]] = {
            'episode_rewards': [], 'episode_lengths': [],
            'avg_delays': [], 'avg_losses': [], 'avg_throughputs': [],
            'qos_violation_rates': [], 'exploration_rates': [],
            'learning_losses': [], 'learning_rates': [],
            'grad_norms': [], 'path_rejections': [],
            'delay_penalties': [], 'completion_rates': [],
        }

        if output_dir is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"training_output_{ts}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Early stopping
        # With eval_interval=25, patience=30 means stop after 750
        # episodes without a new best. That's too aggressive for this
        # task: eval reward fluctuates ±50 just from random flow
        # placement, so a true plateau is hard to distinguish from
        # noise. Bumped to 80 (=2000 episodes between best reward and
        # stop), effectively disabling early stopping for typical
        # 2000-episode runs while still catching truly divergent ones.
        self.best_reward = -np.inf
        self.patience_counter = 0
        self.early_stopping_patience = 80
        self.min_delta = 0.01

        self._save_configuration()
        logger.info(f"DQNTrainer ready. Output: {self.output_dir}")
        logger.info(f"state_dim={self.state_dim}, action_dim={self.action_dim}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _enable_torch_optimizations(self):
        torch.backends.cudnn.benchmark = True
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['OMP_NUM_THREADS'] = str(min(8, (os.cpu_count() or 2) // 2))

    def _save_configuration(self):
        cfg = {
            'device': str(self.device),
            'use_multi_gpu': self.use_multi_gpu,
            'use_amp': self.use_amp,
            'delay_focus': self.delay_focus,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'pytorch_version': torch.__version__,
            'sat_config': self.sat_config.to_dict(),
            'dqn_config': self.dqn_config.to_dict(),
            'sim_config': self.sim_config.to_dict(),
        }
        with open(os.path.join(self.output_dir, 'training_config.json'),
                  'w') as f:
            json.dump(cfg, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Helpers — delay / congestion
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_delay(d) -> float:
        if d is None:
            return 100.0
        try:
            d = float(d)
        except (TypeError, ValueError):
            return 100.0
        if np.isinf(d) or np.isnan(d):
            return 1000.0
        return min(d, 2000.0)

    def _estimate_path_delay_gnn(self, path: List, network_state: Dict[str, Any]
                                 ) -> float:
        """GNN-based path delay estimate (ms).

        Falls back to a deterministic heuristic if the GNN is untrained
        or the query fails.
        """
        if not path or len(path) < 2:
            return 100.0
        try:
            src_idx = self._node_idx_map.get(path[0])
            dst_idx = self._node_idx_map.get(path[-1])
            if src_idx is not None and dst_idx is not None:
                gnn_d = self.agent.predict_path_delay(network_state,
                                                     src_idx, dst_idx)
                if gnn_d > 0.0:
                    # Blend with heuristic to keep early training stable
                    h = self._heuristic_path_delay(path, network_state)
                    return 0.5 * gnn_d + 0.5 * h
        except Exception as e:
            logger.debug(f"GNN delay query failed: {e}")
        return self._heuristic_path_delay(path, network_state)

    @staticmethod
    def _heuristic_path_delay(path: List, network_state: Dict[str, Any]) -> float:
        if not path or len(path) < 2:
            return 100.0
        base_hop = 15.0       # ms
        base = 30.0           # ms (constant overhead)
        cong = 1.0
        utils = network_state.get('link_utilization')
        if utils is not None and len(utils) > 0:
            cong = 1.0 + float(np.mean(utils))
        return base + (len(path) - 1) * base_hop * cong

    @staticmethod
    def _path_congestion(path: List, network_state: Dict[str, Any]) -> float:
        if not path or len(path) < 2:
            return 0.0
        base = min(len(path) / 15.0, 1.0)
        utils = network_state.get('link_utilization')
        if utils is not None and len(utils) > 0:
            base = base * 0.6 + float(np.mean(utils)) * 0.4
        return min(base, 1.0)

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------
    def prepare_state_for_agent(self, network_state: Dict[str, Any],
                                flow_info: Dict[str, Any],
                                update_normalizer: bool = True) -> np.ndarray:
        try:
            paths = self.env.get_k_shortest_paths(
                flow_info['src'], flow_info['dst'], k=self.action_dim)
            network_state['candidate_paths'] = paths
            raw = self._prepare_enhanced_state(network_state, flow_info, paths)
            if update_normalizer:
                self.state_normalizer.update(raw)
            return self.state_normalizer.normalize(raw)
        except Exception as e:
            logger.debug(f"State prep failed: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def _prepare_enhanced_state(self, network_state: Dict[str, Any],
                                flow_info: Dict[str, Any],
                                paths: List[List]) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0

        # 1. Flow characteristics (5 features)
        rate = flow_info.get('demand',
               flow_info.get('rate_mbps',
               flow_info.get('rate', 5e6))) / 1e6   # → Mbps
        state[idx] = min(rate / 100.0, 1.0); idx += 1
        priority = flow_info.get('priority', 2)
        state[idx] = priority / 5.0; idx += 1
        ft = str(flow_info.get('type', 'data')).lower()
        # one-hot: voip / video / other
        state[idx] = 1.0 if ft == 'voip' else 0.0; idx += 1
        state[idx] = 1.0 if ft == 'video' else 0.0; idx += 1
        state[idx] = 1.0 if ft not in ('voip', 'video') else 0.0; idx += 1

        # 2. Network congestion (4 features)
        utils = network_state.get('link_utilization')
        if utils is not None and len(utils) > 0:
            u = np.asarray(utils)
            state[idx] = float(np.percentile(u, 50)); idx += 1
            state[idx] = float(np.percentile(u, 90)); idx += 1
            state[idx] = float(np.mean(u > 0.8)); idx += 1
            state[idx] = float(np.std(u)) if u.size > 1 else 0.0; idx += 1
        else:
            idx += 4

        # 3. Current delays (3 features)
        delays = network_state.get('delays') or network_state.get('link_delays')
        if delays is not None and len(delays) > 0:
            valid = np.asarray([d for d in delays
                                if not np.isinf(d) and not np.isnan(d)])
            if valid.size > 0:
                state[idx]   = float(np.percentile(valid, 50)) / 500.0; idx += 1
                state[idx]   = float(np.max(valid)) / 2000.0;  idx += 1
                state[idx]   = float(np.mean(valid > 500));    idx += 1
            else:
                idx += 3
        else:
            idx += 3

        # 4. Path information — uses the GNN predictor (3 × 4 = 12 features)
        for i in range(3):
            if i < len(paths) and len(paths[i]) >= 2:
                p = paths[i]
                state[idx]     = min(len(p) / 15.0, 1.0)
                pdel = self._estimate_path_delay_gnn(p, network_state)
                state[idx + 1] = min(pdel / 1000.0, 2.0)
                state[idx + 2] = self._path_congestion(p, network_state)
                state[idx + 3] = i / 3.0
            else:
                state[idx]     = 1.0
                state[idx + 1] = 2.0
                state[idx + 2] = 1.0
                state[idx + 3] = i / 3.0
            idx += 4

        # 5. Flow-specific requirements (4 features) — total = 28
        qos_req = self.traffic_config.FLOW_TYPES.get(ft, {})
        max_delay_ms = qos_req.get('delay_req', 200)
        # Strictness: shorter req -> higher strictness
        strictness = max(0.0, 1.0 - max_delay_ms / 1000.0)
        state[idx] = max_delay_ms / 1000.0; idx += 1
        state[idx] = strictness; idx += 1
        state[idx] = min(rate / 100.0, 1.0); idx += 1
        state[idx] = 1.0 if ft in ('voip', 'video', 'gaming', 'control') else 0.0
        idx += 1

        return state

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        """Bounded, delay-focused shaping if env reward is empty."""
        avg_delay = self._safe_delay(info.get('avg_delay', 100.0))
        avg_tput = info.get('avg_throughput', 0)
        active = max(info.get('active_flows', 1), 1)
        qos_v = info.get('qos_violations', 0)

        # Delay component: positive if below 100ms, negative if above 500ms
        if avg_delay < 100:
            delay_c = 0.5
        elif avg_delay < 200:
            delay_c = 0.2
        elif avg_delay < 500:
            delay_c = -0.1 * (avg_delay - 200) / 300.0
        else:
            delay_c = -0.5 - 0.2 * min((avg_delay - 500) / 500.0, 1.0)

        tput_c = 0.0
        if avg_tput > 0:
            tput_c = 0.3 * min(avg_tput / 1e9, 1.0)

        qos_rate = qos_v / active
        qos_c = -0.3 * qos_rate
        success_c = 0.4 * (1.0 - qos_rate)

        r = delay_c + tput_c + qos_c + success_c
        return float(np.clip(r * 5.0, -10.0, 10.0))

    # ------------------------------------------------------------------
    # Per-episode loop
    # ------------------------------------------------------------------
    def train_episode(self, episode_num: int) -> Dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        network_state = self.env.reset()
        # Spatio-temporal GNN keeps a per-node hidden state across snapshots;
        # clear it at each episode boundary so state doesn't leak between
        # episodes. No-op for the simple / full (non-recurrent) models.
        if hasattr(self.gnn_predictor, 'reset_temporal_state'):
            self.gnn_predictor.reset_temporal_state()
        episode_reward = 0.0
        done = False
        step_counter = 0

        # ** Much longer than the previous 20-step cap **
        max_steps = min(self.sim_config.EPISODE_LENGTH, 50)

        ep_metrics = {
            'delays': [], 'losses': [], 'throughputs': [],
            'qos_violations': 0, 'total_flows': 0,
            'step_losses': [], 'path_rejections': 0,
            'completed_flows': 0,
            # Per-step violation rates (averaged at end → real value in [0,1])
            'step_qos_rates': [],
            # Track unique flow IDs we ever saw, and their final status
            'seen_flow_ids': set(),
            # Per-action reward accumulators for diagnostics
            'action_rewards': {0: [], 1: [], 2: []},
        }

        while not done and step_counter < max_steps:
            active_flows = [f for f in self.env.flows
                            if f.get('status', 'active') == 'active']
            if not active_flows:
                break
            flows_to_process = active_flows[:min(5, len(active_flows))]

            routing_decisions = {}
            stored_transitions = []   # (flow_id, state_used, action)

            for flow in flows_to_process:
                try:
                    state = self.prepare_state_for_agent(
                        network_state, flow, update_normalizer=True)
                    action = self.agent.select_action(state, training=True)
                    paths = network_state.get('candidate_paths', [])
                    if not paths:
                        ep_metrics['path_rejections'] += 1
                        continue
                    if action >= len(paths):
                        action = 0  # fall back to shortest
                    routing_decisions[flow['id']] = paths[action]
                    stored_transitions.append((flow['id'], state, action))
                except Exception as e:
                    logger.debug(f"Flow {flow.get('id')} prep error: {e}")
                    continue

            # Environment step
            try:
                next_state_dict, reward, done, info = self.env.step(routing_decisions)
                if abs(reward) < 1e-6:
                    reward = self._calculate_reward(info)
            except Exception as e:
                logger.error(f"env.step error: {e}")
                info = {'avg_delay': 200.0, 'avg_loss': 0.5,
                        'avg_throughput': 0, 'qos_violations': 0,
                        'active_flows': len(active_flows)}
                reward = self._calculate_reward(info)
                next_state_dict = network_state
                done = True

            avg_delay = self._safe_delay(info.get('avg_delay', 100.0))
            flow_rewards = info.get('flow_rewards', {}) or {}

            # Store transitions (re-using the SAME state used for action selection)
            # CRITICAL: credit each flow's transition with its OWN reward.
            # Using the env-averaged reward for every transition was the main
            # cause of the flat reward curve — the agent could not tell
            # which action was responsible for which outcome.
            for flow_id, state, action in stored_transitions:
                if flow_id >= len(self.env.flows):
                    continue
                flow = self.env.flows[flow_id]
                try:
                    next_state = self.prepare_state_for_agent(
                        next_state_dict, flow, update_normalizer=False)
                    flow_delay = self._safe_delay(flow.get('end_to_end_delay', avg_delay))
                    # Prefer per-flow reward; fall back to averaged step reward
                    flow_r = float(flow_rewards.get(flow_id, reward))
                    # Bound to keep Q-values stable
                    flow_r = float(np.clip(flow_r, -10.0, 10.0))
                    self.agent.store_transition(state, action, flow_r,
                                                next_state, done,
                                                delay=flow_delay)
                    # Record per-action reward for diagnostics — the
                    # smoking gun for whether actions are differentiated
                    if action in ep_metrics['action_rewards']:
                        ep_metrics['action_rewards'][action].append(flow_r)
                    ep_metrics['seen_flow_ids'].add(flow_id)
                except Exception as e:
                    logger.debug(f"Store transition error: {e}")

            # Learning step
            if step_counter % self.gradient_accumulation_steps == 0 \
                    and len(self.agent.replay_buffer) >= self.dqn_config.BATCH_SIZE:
                loss = self.agent.learn()
                if loss > 0:
                    ep_metrics['step_losses'].append(loss)

            # Update episode stats
            ep_metrics['delays'].append(info.get('avg_delay', 0))
            ep_metrics['losses'].append(info.get('avg_loss', 0))
            ep_metrics['throughputs'].append(info.get('avg_throughput', 0))
            ep_metrics['qos_violations'] += info.get('qos_violations', 0)
            ep_metrics['step_qos_rates'].append(
                float(info.get('qos_violation_rate', 0.0)))
            ep_metrics['total_flows'] = info.get('active_flows',
                                                 ep_metrics['total_flows'])
            episode_reward += reward
            network_state = next_state_dict
            step_counter += 1

        # End-of-episode
        # Use scheduled epsilon decay when configured; fall back to
        # multiplicative decay otherwise
        if hasattr(self.agent, 'decay_epsilon_for_episode'):
            self.agent.decay_epsilon_for_episode(episode_num)
        else:
            self.agent.update_epsilon()
        # Cosine LR schedule, stepped per episode (set up by
        # set_epsilon_schedule for the same horizon)
        if hasattr(self.agent, 'set_lr_for_episode'):
            self.agent.set_lr_for_episode(episode_num)

        # Aggregate
        delays = [d for d in ep_metrics['delays']
                  if not np.isinf(d) and not np.isnan(d)]
        avg_delay = float(np.mean(delays)) if delays else 100.0
        losses = [l for l in ep_metrics['losses']
                  if not np.isinf(l) and not np.isnan(l)]
        avg_loss = float(np.mean(losses)) if losses else 0.0
        tputs = [t for t in ep_metrics['throughputs']
                 if not np.isinf(t) and not np.isnan(t)]
        avg_tput = float(np.mean(tputs)) if tputs else 0.0

        # Completion / QoS as real proportions in [0, 1].
        #
        # ONLY flows that were actually routed during this episode count
        # toward the denominator. The env spawns 50 flows but the agent
        # routes only ~5 per step → ~45 flows sit untouched with
        # allocated_bandwidth=0, dragging the metric to 0.01-0.03 even
        # when the agent's routing is fine. Old definition was useless.
        served_well = 0
        served_partial = 0
        failed = 0
        touched = 0
        for f in self.env.flows:
            status = f.get('status', 'active')
            if status == 'failed':
                failed += 1
                touched += 1
                continue
            alloc = float(f.get('allocated_bandwidth', 0) or 0)
            # "Touched" = the agent actually routed this flow at some
            # point during the episode (so allocated_bandwidth > 0).
            if alloc <= 0:
                continue
            touched += 1
            d_ms = f.get('end_to_end_delay', float('inf'))
            qos = f.get('qos_requirements', {})
            budget = qos.get('delay_req', 1000)
            demand = max(f.get('demand', 1.0), 1e-6)
            tput_ok = (alloc / demand) > 0.9
            delay_ok = (d_ms < budget) if np.isfinite(d_ms) else False
            if delay_ok and tput_ok:
                served_well += 1
            elif delay_ok or tput_ok:
                served_partial += 1
        denom = max(touched, 1)
        completion_rate = served_well / denom
        failure_rate = failed / denom
        # qos_rate: per-step violation rate averaged over the episode
        per_step_rates = ep_metrics.get('step_qos_rates') or []
        qos_rate = float(np.mean(per_step_rates)) if per_step_rates else \
                   float(np.clip(failure_rate, 0.0, 1.0))

        step_losses = [l for l in ep_metrics['step_losses']
                       if not np.isinf(l) and not np.isnan(l)]
        avg_step_loss = float(np.mean(step_losses)) if step_losses else 0.0
        exploration_rate = self.agent.get_exploration_rate()
        current_lr = self.agent.optimizer.param_groups[0]['lr']

        self.training_metrics['episode_rewards'].append(episode_reward)
        self.training_metrics['episode_lengths'].append(step_counter)
        self.training_metrics['avg_delays'].append(avg_delay)
        self.training_metrics['avg_losses'].append(avg_loss)
        self.training_metrics['avg_throughputs'].append(avg_tput)
        self.training_metrics['qos_violation_rates'].append(qos_rate)
        self.training_metrics['exploration_rates'].append(exploration_rate)
        self.training_metrics['learning_losses'].append(avg_step_loss)
        self.training_metrics['learning_rates'].append(current_lr)
        self.training_metrics['path_rejections'].append(ep_metrics['path_rejections'])
        self.training_metrics['completion_rates'].append(completion_rate)

        # Per-action reward means — flat curve diagnostic.
        # If these are nearly equal it means actions are indistinguishable;
        # the agent literally cannot do better than random.
        action_means: Dict[int, float] = {}
        action_counts: Dict[int, int] = {}
        for a, rs in ep_metrics['action_rewards'].items():
            action_means[a] = float(np.mean(rs)) if rs else 0.0
            action_counts[a] = len(rs)
        if not hasattr(self, '_action_history'):
            self._action_history = {0: [], 1: [], 2: []}
            self._action_count_history = {0: [], 1: [], 2: []}
        for a in self._action_history:
            self._action_history[a].append(action_means.get(a, 0.0))
            self._action_count_history[a].append(action_counts.get(a, 0))

        self.agent.log_performance(episode_reward, avg_delay, qos_rate)

        return {
            'episode': episode_num,
            'reward': episode_reward,
            'length': step_counter,
            'avg_delay': avg_delay,
            'avg_loss': avg_loss,
            'avg_throughput': avg_tput,
            'qos_violation_rate': qos_rate,
            'completion_rate': completion_rate,
            'failure_rate': failure_rate,
            'exploration_rate': exploration_rate,
            'learning_rate': current_lr,
            'step_loss': avg_step_loss,
            'action_means': action_means,
            'action_counts': action_counts,
        }

    # ------------------------------------------------------------------
    # Flat-curve diagnostics
    # ------------------------------------------------------------------
    def _diagnose_flat_curve(self, episode: int, window: int = 20):
        """Surface common causes of a flat reward curve.
        """
        rewards = self.training_metrics['episode_rewards']
        if len(rewards) < window:
            return

        recent = np.asarray(rewards[-window:], dtype=float)
        r_std = float(np.std(recent))
        r_mean = float(np.mean(recent))
        # Detrended std: if reward is steadily growing the std is high
        # even when the curve isn't flat.
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        detrend = recent - (slope * np.arange(len(recent)) + r_mean)
        d_std = float(np.std(detrend))
        # Signal-to-noise: a clear upward trend needs |slope * window| ~ r_std
        snr = abs(slope) * len(recent) / max(r_std, 1e-6)

        explore = self.training_metrics['exploration_rates'][-1] \
            if self.training_metrics['exploration_rates'] else 0.0
        losses = self.training_metrics['learning_losses'][-window:]
        l_recent = float(np.mean(losses)) if losses else 0.0

        msgs: List[str] = []
        # Trend-free is the most common "I'm learning nothing" failure
        if snr < 0.3:
            msgs.append(
                f"reward has no trend (slope={slope:+.4f}/ep, std={r_std:.3f}, "
                f"trend/noise={snr:.2f}) — agent is not improving"
            )
        elif abs(slope) > 0.005:
            msgs.append(
                f"reward IS trending: slope={slope:+.4f}/ep "
                f"(trend/noise={snr:.2f}) — keep training"
            )
        if l_recent > 0 and snr < 0.3:
            msgs.append(f"loss is moving ({l_recent:.4f}) but reward isn't — "
                        "probably overfitting stale buffer or action collapse")
        if explore < 0.01:
            msgs.append("exploration ~0: agent is fully greedy "
                        "(might be stuck on one action) — temporarily raise epsilon")
        elif explore > 0.95:
            msgs.append("exploration ~1: agent is still nearly random — "
                        "wait for epsilon decay or lower epsilon_min")

        # Inspect candidate-path diversity for a sample flow
        try:
            state = self.env.get_state() if hasattr(self.env, 'get_state') else {}
            paths = state.get('candidate_paths') or []
            if len(paths) >= 2:
                lengths = [len(p) for p in paths]
                # Jaccard overlap between path edge sets
                def _edges(p):
                    return set((min(str(p[i]), str(p[i + 1])),
                                max(str(p[i]), str(p[i + 1])))
                               for i in range(len(p) - 1))
                e0 = _edges(paths[0])
                overlaps = []
                for p in paths[1:]:
                    ep = _edges(p)
                    if e0 or ep:
                        overlaps.append(len(e0 & ep) / max(len(e0 | ep), 1))
                if overlaps and max(overlaps) > 0.75:
                    msgs.append(
                        f"candidate paths overlap {max(overlaps):.0%} on edges "
                        f"(lengths={lengths}) — actions are barely distinguishable"
                    )
        except Exception:
            pass

        if msgs:
            logger.warning(f"[diagnose ep {episode}] " + "; ".join(msgs))
        else:
            logger.info(
                f"[diagnose ep {episode}] reward std={r_std:.3f} "
                f"slope={slope:+.4f}/ep loss={l_recent:.4f} eps={explore:.2f} — looks healthy")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, num_episodes: int = 200, save_interval: int = 50,
              eval_interval: int = 25):
        logger.info(f"Training for {num_episodes} episodes")

        # Compute the random-policy baseline ONCE up front. Every later
        # eval reward should be compared to this number — that's the
        # actual signal of how much the agent has learned.
        try:
            random_baseline = self.evaluate_random_baseline(num_episodes=3)
            self.random_baseline = random_baseline
            logger.info(
                f"Random-policy baseline: r={random_baseline:.3f}  "
                f"(agent eval > this = learned policy is better than random)")
        except Exception as e:
            logger.warning(f"Random-baseline eval failed: {e}")
            self.random_baseline = 0.0

        # Tell the agent how many episodes we have so it can schedule a
        # sensible epsilon decay. Without this, eps stays near 1.0 for
        # short runs and you only ever see the random-policy reward.
        if hasattr(self.agent, 'set_epsilon_schedule'):
            self.agent.set_epsilon_schedule(num_episodes, decay_fraction=0.7)
        # Re-assert fixed LR (set_epsilon_schedule also configures the cosine
        # horizon; in fixed mode we override it back to the constant rate).
        if self.fixed_lr is not None and hasattr(self.agent, 'set_fixed_lr'):
            self.agent.set_fixed_lr(self.fixed_lr)
        t0 = time.time()
        ep_times: List[float] = []

        for ep in range(num_episodes):
            t_ep = time.time()
            m = self.train_episode(ep)
            ep_times.append(time.time() - t_ep)

            avg_t = float(np.mean(ep_times[-5:]))
            # Surface link utilization so it's obvious when the network
            # is too lightly loaded for the agent to learn anything
            net_stats = {}
            try:
                net_stats = self.env.get_network_stats()
            except Exception:
                pass
            util = net_stats.get('avg_utilization', 0.0)
            am = m.get('action_means', {})
            # Spread between action means is the single best signal that
            # actions are distinguishable. If this is ~0 the agent
            # literally cannot do better than random.
            spread = (max(am.values()) - min(am.values())) if am else 0.0
            logger.info(
                f"Ep {ep+1}/{num_episodes} | t={ep_times[-1]:.1f}s "
                f"(avg {avg_t:.1f}s) | r={m['reward']:.3f} | "
                f"delay={m['avg_delay']:.2f}ms | "
                f"completion={m['completion_rate']:.2f} | "
                f"util={util:.2f} | eps={self.agent.epsilon:.3f} | "
                f"a_spread={spread:.2f}")

            # Diagnostic check every 20 episodes — surfaces flat curve
            # causes early so you don't have to wait until training ends
            if (ep + 1) % 20 == 0:
                self._diagnose_flat_curve(ep + 1)

            if eval_interval > 0 and (ep + 1) % eval_interval == 0:
                try:
                    eval_m = self.evaluate(num_episodes=2)
                    eval_r = eval_m.get('rewards', 0.0)
                    if eval_r > self.best_reward + self.min_delta:
                        self.best_reward = eval_r
                        self.patience_counter = 0
                        self.save_checkpoint(ep + 1, best=True)
                        logger.info(f"** new best reward: {eval_r:.3f} **")
                    else:
                        self.patience_counter += 1
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")

            if save_interval > 0 and (ep + 1) % save_interval == 0:
                self.save_checkpoint(ep + 1)
                self.plot_training_metrics()

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping @ episode {ep+1}")
                break

        total_time = time.time() - t0
        logger.info(f"Training done in {total_time:.1f}s")
        self.save_checkpoint(num_episodes, final=True)
        self.plot_training_metrics()
        self._create_training_summary(total_time)
        return self.training_metrics

    # ------------------------------------------------------------------
    # Saving / plotting / summary
    # ------------------------------------------------------------------
    def _create_training_summary(self, training_time: float):
        rewards = self.training_metrics['episode_rewards']
        delays = self.training_metrics['avg_delays']
        if not rewards:
            return
        summary = {
            'total_training_time_seconds': training_time,
            'total_episodes': len(rewards),
            'final_reward': float(rewards[-1]),
            'final_delay': float(delays[-1]) if delays else 0.0,
            'average_reward': float(np.mean(rewards)),
            'average_delay': float(np.mean(delays)) if delays else 0.0,
            'best_reward': float(np.max(rewards)),
            'best_delay': float(np.min(delays)) if delays else 0.0,
            'best_completion_rate': float(np.max(self.training_metrics['completion_rates'])) if self.training_metrics['completion_rates'] else 0.0,
        }
        with open(os.path.join(self.output_dir, 'training_summary.json'),
                  'w') as f:
            json.dump(summary, f, indent=2)

    def save_checkpoint(self, episode: int, best: bool = False,
                        final: bool = False):
        name = "final" if final else ("best" if best else f"checkpoint_episode_{episode}")
        d = os.path.join(self.output_dir, 'checkpoints', name)
        os.makedirs(d, exist_ok=True)

        torch.save(self.agent.q_network.state_dict(),
                   os.path.join(d, "dqn_model.pth"))
        torch.save(self.agent.target_network.state_dict(),
                   os.path.join(d, "target_model.pth"))

        # Save tail of each metric (last 200)
        ms = {k: v[-200:] if isinstance(v, list) else v
              for k, v in self.training_metrics.items()}
        with open(os.path.join(d, "metrics.json"), 'w') as f:
            json.dump(ms, f, indent=2)
        logger.info(f"Checkpoint saved: {name}")

    def plot_training_metrics(self):
        rewards = [r for r in self.training_metrics['episode_rewards']
                   if not np.isinf(r) and not np.isnan(r)]
        delays = [d for d in self.training_metrics['avg_delays']
                  if not np.isinf(d) and not np.isnan(d)]
        if not rewards:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(rewards, alpha=0.6, linewidth=1)
        if len(rewards) > 10:
            w = min(20, len(rewards) // 2)
            ma = np.convolve(rewards, np.ones(w)/w, mode='valid')
            axes[0, 0].plot(range(w-1, len(rewards)), ma, 'r-', linewidth=2,
                            label=f'MA({w})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        if delays:
            axes[0, 1].plot(delays, alpha=0.7, color='orange', linewidth=1)
            axes[0, 1].set_title('Avg Delay (ms)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].grid(True, alpha=0.3)

        expl = [e for e in self.training_metrics['exploration_rates']
                if not np.isinf(e) and not np.isnan(e)]
        if expl:
            axes[1, 0].plot(expl, alpha=0.7, color='green', linewidth=1)
            axes[1, 0].set_title('Exploration rate')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].grid(True, alpha=0.3)

        ll = [l for l in self.training_metrics['learning_losses']
              if not np.isinf(l) and not np.isnan(l) and l > 0]
        if ll:
            axes[1, 1].plot(ll, alpha=0.7, color='brown', linewidth=1)
            axes[1, 1].set_title('DQN learning loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f'Training metrics — {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_random_baseline(self, num_episodes: int = 2) -> float:
        """Evaluate with random action selection — the "do-nothing" baseline.
        """
        total_r = 0.0
        for _ in range(num_episodes):
            network_state = self.env.reset()
            if hasattr(self.gnn_predictor, 'reset_temporal_state'):
                self.gnn_predictor.reset_temporal_state()
            ep_reward = 0.0
            done = False
            sc = 0
            max_eval_steps = min(self.sim_config.EPISODE_LENGTH, 25)
            while not done and sc < max_eval_steps:
                active = [f for f in self.env.flows
                          if f.get('status', 'active') == 'active']
                flows = active[:5]
                routing = {}
                for flow in flows:
                    paths = network_state.get('candidate_paths', [])
                    if not paths:
                        try:
                            paths = self.env.get_k_shortest_paths(
                                flow['src'], flow['dst'], k=self.action_dim)
                        except Exception:
                            paths = []
                    if paths:
                        a = random.randint(0, len(paths) - 1)
                        routing[flow['id']] = paths[a]
                try:
                    network_state, r, done, _ = self.env.step(routing)
                except Exception:
                    r = -1.0
                    done = True
                ep_reward += r
                sc += 1
            total_r += ep_reward
        return total_r / max(num_episodes, 1)

    def evaluate(self, num_episodes: int = 3, epsilon: float = 0.05
                 ) -> Dict[str, float]:
        logger.info(f"Evaluating for {num_episodes} episodes...")
        eval_m = {'rewards': [], 'delays': [], 'throughputs': [],
                  'success_rates': [], 'episode_times': []}
        self.agent.q_network.eval()

        for ep in range(num_episodes):
            t0 = time.time()
            network_state = self.env.reset()
            if hasattr(self.gnn_predictor, 'reset_temporal_state'):
                self.gnn_predictor.reset_temporal_state()
            ep_reward = 0.0
            done = False
            sc = 0
            max_eval_steps = min(self.sim_config.EPISODE_LENGTH, 25)
            info = {}

            while not done and sc < max_eval_steps:
                active = [f for f in self.env.flows
                          if f.get('status', 'active') == 'active']
                flows = active[:5]
                routing = {}
                for flow in flows:
                    try:
                        state = self.prepare_state_for_agent(
                            network_state, flow, update_normalizer=False)
                        a = self.agent.select_action(state, training=False,
                                                     epsilon=epsilon)
                        paths = network_state.get('candidate_paths', [])
                        if paths and a < len(paths):
                            routing[flow['id']] = paths[a]
                    except Exception:
                        continue
                try:
                    network_state, r, done, info = self.env.step(routing)
                    if abs(r) < 1e-6:
                        r = self._calculate_reward(info)
                except Exception:
                    r = -1.0
                    done = True
                    info = {'avg_delay': 200, 'qos_violations': 0,
                            'active_flows': len(active)}
                ep_reward += r
                sc += 1

            avg_delay = self._safe_delay(info.get('avg_delay', 100.0))
            avg_tput = info.get('avg_throughput', 0)
            active = info.get('active_flows', 0)
            qos_v = info.get('qos_violations', 0)
            success_rate = max(0.0, (active - qos_v) / max(active, 1))

            eval_m['rewards'].append(ep_reward)
            eval_m['delays'].append(avg_delay)
            eval_m['throughputs'].append(avg_tput)
            eval_m['success_rates'].append(success_rate)
            eval_m['episode_times'].append(time.time() - t0)
            logger.info(f"Eval {ep+1}: r={ep_reward:.3f} "
                        f"delay={avg_delay:.1f}ms success={success_rate:.2f}")

        self.agent.q_network.train()

        averages = {}
        for k, v in eval_m.items():
            clean = [x for x in v if not np.isinf(x) and not np.isnan(x)]
            averages[k] = float(np.mean(clean)) if clean else 0.0

        with open(os.path.join(self.output_dir, 'evaluation_results.json'),
                  'w') as f:
            json.dump(averages, f, indent=2)
        # Show the gap over random — the actual signal of learning
        baseline = getattr(self, 'random_baseline', 0.0)
        gap = averages['rewards'] - baseline
        logger.info(f"Eval: r={averages['rewards']:.3f} "
                    f"(vs random {baseline:+.3f}, gap {gap:+.3f}) "
                    f"d={averages['delays']:.1f}ms "
                    f"success={averages['success_rates']:.3f}")
        return averages


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------
def create_and_train_model(num_episodes: int = 100, use_debug: bool = True,
                           gnn_model_type: str = 'simple',
                           gnn_hidden_dim: int = 64,
                           output_dir: Optional[str] = None):
    from config import OptimizedConfigs
    if use_debug:
        sat, traffic, dqn, sim = OptimizedConfigs.get_debug_config()
    else:
        sat, traffic, dqn, sim = OptimizedConfigs.get_training_config()

    trainer = DQNTrainer(sat, traffic, dqn, sim,
                         use_multi_gpu=False,
                         use_amp=True,
                         gradient_accumulation_steps=4,
                         delay_focus=True,
                         output_dir=output_dir,
                         gnn_model_type=gnn_model_type,
                         gnn_hidden_dim=gnn_hidden_dim)
    metrics = trainer.train(num_episodes=num_episodes,
                            save_interval=25,
                            eval_interval=25)
    eval_m = trainer.evaluate(num_episodes=3)
    return trainer, metrics, eval_m


def main():
    parser = argparse.ArgumentParser(description='Train DQN for satellite routing')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='debug',
                        choices=['debug', 'training', 'production'])
    parser.add_argument('--gnn-model', type=str, default='simple',
                        choices=['simple', 'full', 'spatiotemporal'],
                        help="GNN delay-predictor architecture: 'simple' "
                             "(SimpleGNN), 'full' (multi-head attention), or "
                             "'spatiotemporal' (attention + GRU recurrence).")
    parser.add_argument('--gnn-hidden', type=int, default=64,
                        help='GNN hidden dimension (simple=64; full/ST use ≥128).')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"PyTorch {torch.__version__}  cuda={torch.cuda.is_available()}")

    trainer, metrics, eval_m = create_and_train_model(
        num_episodes=args.episodes,
        use_debug=(args.config == 'debug'),
        gnn_model_type=args.gnn_model,
        gnn_hidden_dim=args.gnn_hidden)

    print('\n' + '=' * 80)
    print('TRAINING COMPLETED')
    print('=' * 80)
    rewards = metrics['episode_rewards']
    delays = metrics['avg_delays']
    if rewards:
        print(f"Episodes:               {len(rewards)}")
        print(f"Final episode reward:   {rewards[-1]:.3f}")
        print(f"Final episode delay:    {delays[-1]:.2f} ms")
        print(f"Best reward:            {max(rewards):.3f}")
        print(f"Best (min) delay:       {min(delays):.2f} ms")
        print(f"Eval success rate:      {eval_m.get('success_rates', 0):.3f}")
        print(f"Output:                 {trainer.output_dir}")

    try:
        from visulaize import MetricsVisualizer
        viz = MetricsVisualizer(output_dir=os.path.join(trainer.output_dir,
                                                        "plots"))
        viz.plot_all(trainer.training_metrics)
    except ImportError:
        try:
            from visualize import MetricsVisualizer  # alternate name
            viz = MetricsVisualizer(output_dir=os.path.join(trainer.output_dir,
                                                            "plots"))
            viz.plot_all(trainer.training_metrics)
        except ImportError:
            logger.info("MetricsVisualizer not available, skipping extra plots")


if __name__ == "__main__":
    main()
