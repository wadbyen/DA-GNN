"""
utils.py - Visualization, logging and IO helpers.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Choose a non-interactive backend when DISPLAY is unavailable. This must
# happen before pyplot is imported.
import matplotlib
if not os.environ.get('DISPLAY') and matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx            # noqa: E402

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging / IO
# ---------------------------------------------------------------------------
def _ensure_dir(filename: str) -> None:
    """Create the parent directory of `filename` if needed (no-op for bare names)."""
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
  
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        _ensure_dir(log_file)
        handlers.append(logging.FileHandler(log_file))

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )
    logger.info(f"Logging setup complete. Level: {log_level}")


def save_results(results: Dict[str, Any], filename: str):
    """Persist a results dict. Format inferred from extension."""
    _ensure_dir(filename)

    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)
    elif filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(filename, 'w') as f:
            f.write(str(results))

    logger.info(f"Results saved to {filename}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load a previously-saved results dict.
    """
    if not os.path.exists(filename):
        logger.warning(f"File {filename} does not exist")
        return {}

    try:
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        # default: try JSON
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {filename}: {e}")
        return {}


def _json_default(obj):
    """JSON fallback encoder for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Network / traffic visualizations
# ---------------------------------------------------------------------------
def _circular_positions(graph: nx.Graph):
    """Compute simple circular positions for satellites + ground stations."""
    pos: Dict[Any, tuple] = {}
    node_types = nx.get_node_attributes(graph, 'type')

    satellite_nodes = [n for n in graph.nodes() if node_types.get(n) == 'satellite']
    ground_nodes = [n for n in graph.nodes() if node_types.get(n) == 'ground_station']

    if satellite_nodes:
        angle = 2 * np.pi / len(satellite_nodes)
        for i, node in enumerate(satellite_nodes):
            pos[node] = (np.cos(i * angle), np.sin(i * angle))

    if ground_nodes:
        angle = 2 * np.pi / len(ground_nodes)
        for i, node in enumerate(ground_nodes):
            pos[node] = (1.3 * np.cos(i * angle), 1.3 * np.sin(i * angle))

    # Any nodes that still have no position get one via spring layout
    missing = [n for n in graph.nodes() if n not in pos]
    if missing:
        # restrict to subgraph and merge
        spring = nx.spring_layout(graph.subgraph(missing), seed=0)
        pos.update(spring)
    return pos


def visualize_network(graph: nx.Graph,
                      node_colors: Optional[Dict] = None,
                      edge_colors: Optional[Dict] = None,
                      title: str = "Satellite Network",
                      save_path: Optional[str] = None):
    """Draw the satellite network."""
    plt.figure(figsize=(12, 10))
    pos = _circular_positions(graph)
    node_types = nx.get_node_attributes(graph, 'type')

    # Node colours
    colour_for = lambda n: 'lightblue' if node_types.get(n) == 'satellite' else 'lightgreen'
    nx.draw_networkx_nodes(graph, pos,
                           node_color=[colour_for(n) for n in graph.nodes()],
                           node_size=100, alpha=0.8)

    # Edges grouped by type
    edge_groups = {
        'intra_orbit': ('blue', 1, 0.5),
        'inter_orbit': ('green', 1, 0.5),
        'cross_seam':  ('red', 1, 0.5),
        'sgl':         ('orange', 2, 0.7),
    }
    for etype, (colour, width, alpha) in edge_groups.items():
        edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') == etype]
        if edges:
            nx.draw_networkx_edges(graph, pos, edgelist=edges,
                                   edge_color=colour, width=width, alpha=alpha)

    nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue',   lw=2, label='Intra-orbit ISL'),
        Line2D([0], [0], color='green',  lw=2, label='Inter-orbit ISL'),
        Line2D([0], [0], color='red',    lw=2, label='Cross-seam ISL'),
        Line2D([0], [0], color='orange', lw=2, label='Satellite-Ground Link'),
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='lightblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ground Station',
               markerfacecolor='lightgreen', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Network visualization saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


def visualize_traffic(traffic_matrix: np.ndarray,
                      node_labels: Optional[List[str]] = None,
                      title: str = "Traffic Matrix",
                      save_path: Optional[str] = None):
    """Heatmap of a traffic matrix."""
    plt.figure(figsize=(12, 10))

    if HAS_SEABORN:
        sns.heatmap(traffic_matrix, cmap='YlOrRd',
                    xticklabels=node_labels, yticklabels=node_labels,
                    cbar_kws={'label': 'Traffic Intensity'})
    else:
        plt.imshow(traffic_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Traffic Intensity')
        if node_labels:
            plt.xticks(range(len(node_labels)), node_labels, rotation=90)
            plt.yticks(range(len(node_labels)), node_labels)

    plt.title(title)
    plt.xlabel('Destination')
    plt.ylabel('Source')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Traffic visualization saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


def visualize_path(graph: nx.Graph, path: List,
                   node_colors: Optional[Dict] = None,
                   title: str = "Selected Path",
                   save_path: Optional[str] = None):
    """Highlight a single routing path on the network."""
    plt.figure(figsize=(12, 10))
    pos = _circular_positions(graph)
    node_types = nx.get_node_attributes(graph, 'type')

    path_set = set(path)

    def _node_color(n):
        if n in path_set:
            return 'red' if node_types.get(n) == 'satellite' else 'darkred'
        return 'lightblue' if node_types.get(n) == 'satellite' else 'lightgreen'

    nx.draw_networkx_nodes(graph, pos,
                           node_color=[_node_color(n) for n in graph.nodes()],
                           node_size=100, alpha=0.8)

    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5)

    if len(path) >= 2:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                               edge_color='red', width=3, alpha=0.8)

    path_labels = {node: str(node) for node in path}
    nx.draw_networkx_labels(graph, pos, labels=path_labels,
                            font_size=10, font_weight='bold')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Path visualization saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


def plot_performance_comparison(baseline_metrics: Dict[str, List],
                                dqn_metrics: Dict[str, List],
                                metric_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
    """Plot baseline vs DQN curves side by side."""
    if metric_names is None:
        metric_names = ['rewards', 'delays', 'qos_violation_rates']

    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    titles = {
        'rewards': ('Episode Rewards', 'Reward'),
        'delays': ('Average Delay', 'Delay (ms)'),
        'qos_violation_rates': ('QoS Violation Rate', 'Violation Rate'),
        'throughputs': ('Average Throughput', 'Throughput (bps)'),
    }

    for ax, metric in zip(axes, metric_names):
        if baseline_metrics.get(metric):
            ax.plot(baseline_metrics[metric], 'b-', label='Baseline', alpha=0.7)
        if dqn_metrics.get(metric):
            ax.plot(dqn_metrics[metric], 'r-', label='DQN', alpha=0.7)

        title, ylabel = titles.get(metric, (metric, metric))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if metric == 'throughputs':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance comparison saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reports / stats
# ---------------------------------------------------------------------------
def create_metrics_report(training_metrics: Dict[str, List],
                          evaluation_metrics: Dict[str, float],
                          config: Dict[str, Any],
                          save_path: Optional[str] = None) -> str:
    """Produce a human-readable summary of a training run."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("SATELLITE NETWORK ROUTING OPTIMIZATION - TRAINING REPORT")
    lines.append("=" * 80)
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("TRAINING STATISTICS")
    lines.append("-" * 40)
    rewards = training_metrics.get('episode_rewards') or []
    if rewards:
        lines.append(f"Total episodes trained: {len(rewards)}")
        recent = rewards[-100:] if len(rewards) >= 100 else rewards
        lines.append(f"Average reward (last {len(recent)} eps): {np.mean(recent):.3f}")
        recent_delays = (training_metrics.get('avg_delays') or [])[-len(recent):]
        if recent_delays:
            lines.append(f"Average delay (last {len(recent)} eps): {np.mean(recent_delays):.2f} ms")
        recent_qos = (training_metrics.get('qos_violation_rates') or [])[-len(recent):]
        if recent_qos:
            lines.append(f"Average QoS violation rate (last {len(recent)} eps): {np.mean(recent_qos):.3f}")

    lines.append("")
    lines.append("EVALUATION STATISTICS")
    lines.append("-" * 40)
    fmt = {
        'rewards': ('Average reward', '{:.3f}'),
        'delays': ('Average delay', '{:.2f} ms'),
        'losses': ('Average loss', '{:.4f}'),
        'throughputs': ('Average throughput', '{:.2e} bps'),
        'qos_violations': ('Average QoS violations', '{:.2f}'),
        'success_rates': ('Success rate', '{:.3f}'),
    }
    for key, value in evaluation_metrics.items():
        if key in fmt:
            label, spec = fmt[key]
            try:
                lines.append(f"{label}: {spec.format(float(value))}")
            except Exception:
                lines.append(f"{label}: {value}")

    lines.append("")
    lines.append("CONFIGURATION SUMMARY")
    lines.append("-" * 40)
    sat_conf = config.get('satellite_config', {})
    if sat_conf:
        lines.append(f"Satellites: {sat_conf.get('NUM_SATELLITES', 'N/A')}")
        lines.append(f"Orbits: {sat_conf.get('NUM_ORBITS', 'N/A')}")
        lines.append(f"Altitude: {sat_conf.get('ALTITUDE', 'N/A')} km")
        isl = sat_conf.get('ISL_BANDWIDTH', None)
        if isinstance(isl, (int, float)):
            lines.append(f"ISL Bandwidth: {isl:.0f} Hz")
    dqn_conf = config.get('dqn_config', {})
    if dqn_conf:
        lines.append(f"Learning rate: {dqn_conf.get('LR', 'N/A')}")
        lines.append(f"Gamma: {dqn_conf.get('GAMMA', 'N/A')}")
        lines.append(f"Buffer size: {dqn_conf.get('BUFFER_SIZE', 'N/A')}")

    lines.append("")
    lines.append("PERFORMANCE INSIGHTS")
    lines.append("-" * 40)
    sr = float(evaluation_metrics.get('success_rates', 0) or 0)
    if sr > 0.9:
        lines.append("[OK] Excellent performance: Success rate > 90%")
    elif sr > 0.7:
        lines.append("[OK] Good performance: Success rate > 70%")
    else:
        lines.append("[WARN] Needs improvement: Success rate < 70%")

    delay = float(evaluation_metrics.get('delays', 1000) or 1000)
    if delay < 100:
        lines.append("[OK] Low latency: Average delay < 100 ms")
    elif delay < 200:
        lines.append("[OK] Acceptable latency: Average delay < 200 ms")
    else:
        lines.append("[WARN] High latency: Average delay > 200 ms")

    lines.append("")
    lines.append("=" * 80)
    report = "\n".join(lines)

    if save_path:
        _ensure_dir(save_path)
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Metrics report saved to {save_path}")
    return report


def calculate_statistical_metrics(data: List[float]) -> Dict[str, float]:
    """Return a dict of summary statistics for a 1-D sequence."""
    if not data:
        return {}
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(q3 - q1),
    }


def save_training_history(history: Dict[str, Any], filename: str):
    """Serialize a training history dict to JSON."""
    _ensure_dir(filename)
    serial = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            serial[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            serial[key] = [v.tolist() for v in value]
        else:
            serial[key] = value
    with open(filename, 'w') as f:
        json.dump(serial, f, indent=2, default=_json_default)
    logger.info(f"Training history saved to {filename}")


def load_training_history(filename: str) -> Dict[str, Any]:
    """Load training history; numeric lists become numpy arrays."""
    try:
        with open(filename, 'r') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training history from {filename}: {e}")
        return {}

    for key, value in history.items():
        if isinstance(value, list) and value:
            if all(isinstance(v, (int, float)) for v in value):
                history[key] = np.array(value)
            elif isinstance(value[0], list) and value[0] and \
                    all(isinstance(v, (int, float)) for v in value[0]):
                history[key] = np.array(value)
    return history


def plot_convergence_analysis(loss_history: List[float],
                              reward_history: List[float],
                              window_size: int = 100,
                              save_path: Optional[str] = None):
    """Loss & reward convergence + histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(loss_history, alpha=0.5)
    if len(loss_history) > window_size:
        ma = np.convolve(loss_history, np.ones(window_size) / window_size, mode='valid')
        axes[0, 0].plot(range(window_size - 1, len(loss_history)), ma, 'r-', linewidth=2)
    axes[0, 0].set_title('Loss Convergence')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    if loss_history and min(loss_history) > 0:
        axes[0, 0].set_yscale('log')

    # Reward
    axes[0, 1].plot(reward_history, alpha=0.5)
    if len(reward_history) > window_size:
        ma = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
        axes[0, 1].plot(range(window_size - 1, len(reward_history)), ma, 'r-', linewidth=2)
    axes[0, 1].set_title('Reward Convergence')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)

    if loss_history:
        axes[1, 0].hist(loss_history, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True)

    if reward_history:
        axes[1, 1].hist(reward_history, bins=50, alpha=0.7, density=True)
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence analysis saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass
