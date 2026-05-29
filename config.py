"""
config.py - Configuration classes for satellite network, DQN, and traffic patterns
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# 1. Satellite / constellation parameters
# ---------------------------------------------------------------------------
@dataclass
class SatelliteConfig:
    """Starlink-like satellite constellation configuration"""
    # Constellation
    NUM_SATELLITES: int = 1584
    NUM_ORBITS: int = 72
    SATELLITES_PER_ORBIT: int = 22
    ALTITUDE: float = 550        # km
    EARTH_RADIUS: float = 6371   # km
    ORBITAL_INCLINATION: float = 86.4  # degrees

    # Bandwidths
    SATELLITE_BANDWIDTH: float = 20e9   # 20 Gbps per satellite
    ISL_BANDWIDTH: float = 10e9         # 10 Gbps per ISL
    GSL_BANDWIDTH: float = 5e9          # 5 Gbps per GSL

    # Traffic
    PACKET_SIZE: float = 1e6                # bits  (1 Mbit)
    MAX_PACKETS_PER_SECOND: float = 10000

    # Topology
    GROUND_STATIONS: int = 50
    PROPAGATION_SPEED: float = 3e5          # km/s
    MAX_ISL_PER_SAT: int = 4
    MAX_GSL_PER_SAT: int = 2

    # Congestion
    MAX_QUEUE_SIZE_PACKETS: int = 1000
    MIN_LINK_UTILIZATION: float = 0.1
    MAX_LINK_UTILIZATION: float = 0.85
    CRITICAL_LINK_UTILIZATION: float = 0.95

    # QoS
    MAX_ALLOWED_DELAY_MS: float = 200.0
    MAX_ALLOWED_LOSS_RATE: float = 0.01

    # Scaling
    SCALING_FACTOR: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}

    def calculate_max_throughput(self) -> float:
        return self.SATELLITE_BANDWIDTH * 0.7

    def calculate_propagation_delay(self, distance_km: float) -> float:
        """Propagation delay in **milliseconds**"""
        return (distance_km / self.PROPAGATION_SPEED) * 1000.0


# ---------------------------------------------------------------------------
# 2. Traffic configuration
# ---------------------------------------------------------------------------
@dataclass
class TrafficConfig:
    """Traffic patterns and QoS requirements.
    """

    class FlowType(Enum):
        VIDEO = "video"
        VOIP = "voip"
        WEB = "web"
        FILE_TRANSFER = "file_transfer"
        GAMING = "gaming"
        CONTROL = "control"

    # String-keyed flow specifications  (FIX)
    FLOW_TYPES: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'video': {
            'delay_req': 150,  'loss_req': 1e-3, 'bw_req': 5e6,
            'priority': 3,    'burst_factor': 1.5,
            'min_bw': 1e6,    'max_bw': 20e6,
        },
        'voip': {
            'delay_req': 50,   'loss_req': 1e-4, 'bw_req': 64e3,
            'priority': 4,     'burst_factor': 1.1,
            'min_bw': 32e3,    'max_bw': 128e3,
        },
        'web': {
            'delay_req': 100,  'loss_req': 1e-3, 'bw_req': 2e6,
            'priority': 2,     'burst_factor': 2.0,
            'min_bw': 500e3,   'max_bw': 10e6,
        },
        'file_transfer': {
            'delay_req': 1000, 'loss_req': 1e-3, 'bw_req': 10e6,
            'priority': 1,     'burst_factor': 1.2,
            'min_bw': 1e6,     'max_bw': 100e6,
        },
        'gaming': {
            'delay_req': 30,   'loss_req': 1e-3, 'bw_req': 5e6,
            'priority': 4,     'burst_factor': 1.3,
            'min_bw': 2e6,     'max_bw': 20e6,
        },
        'control': {
            'delay_req': 10,   'loss_req': 1e-5, 'bw_req': 10e3,
            'priority': 5,     'burst_factor': 1.0,
            'min_bw': 1e3,     'max_bw': 100e3,
        },
        # ``data`` is kept as an alias for backward compatibility with
        # parts of the code that still use the legacy name.
        'data': {
            'delay_req': 200,  'loss_req': 1e-3, 'bw_req': 5e6,
            'priority': 2,     'burst_factor': 1.5,
            'min_bw': 1e6,     'max_bw': 50e6,
        },
    })

    TRAFFIC_PATTERNS: List[str] = field(default_factory=lambda:
        ['uniform', 'bursty', 'diurnal', 'spiky'])

    BACKGROUND_TRAFFIC_RATIO: float = 0.2

    # Congestion-control thresholds
    CONGESTION_DETECTION_THRESHOLD: float = 0.8
    CONGESTION_RECOVERY_THRESHOLD: float = 0.6

    # Flow admission
    MAX_CONCURRENT_FLOWS: int = 100
    FLOW_ADMISSION_THRESHOLD: float = 0.7

    # Traffic shaping
    TOKEN_BUCKET_RATE: float = 100e6
    TOKEN_BUCKET_SIZE: float = 1000

    def get_flow_weights(self) -> List[float]:
        """Weights for choosing a flow type at random.

        Order must match the iteration order of ``FLOW_TYPES``.
        """
        return [0.25, 0.15, 0.30, 0.15, 0.10, 0.05, 0.00]

    def get_flow_admission_probability(self, current_utilization: float) -> float:
        if current_utilization < 0.5:
            return 1.0
        elif current_utilization < 0.8:
            return 1.0 - (current_utilization - 0.5) / 0.3
        else:
            return 0.2


# ---------------------------------------------------------------------------
# 3. DQN hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    """DQN hyperparameters - reconciled with trainer dimensions."""
    # *** Architecture - aligned with _prepare_enhanced_state output ***
    STATE_DIM: int = 28
    ACTION_DIM: int = 5
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [256, 128, 64])
    BUFFER_SIZE: int = 30000
    BATCH_SIZE: int = 64
    LEARNING_STARTS: int = 250

    # Learning
    GAMMA: float = 0.99
    LR: float = 5e-4
    LR_MIN: float = 1e-5
    LR_MAX: float = 1e-3
    TAU: float = 0.005             # soft update; slightly more aggressive

    EPS_START: float = 1.0

    EPS_END: float = 0.02
    EPS_DECAY: float = 0.985
    EPS_DECAY_STEPS: int = 5000

    # Training schedule
    UPDATE_EVERY: int = 4
    TARGET_UPDATE: int = 500
    GRADIENT_CLIP: float = 1.0

    # Advanced techniques
    USE_DOUBLE_DQN: bool = True
    USE_DUELING: bool = True
    USE_PER: bool = True
    USE_NOISY_NETS: bool = False

    # PER parameters
    PER_ALPHA: float = 0.6
    PER_BETA: float = 0.4
    PER_BETA_INCREMENT: float = 0.001

    # Multi-step learning
    N_STEP: int = 3

    # Dynamic adjustment
    PERFORMANCE_WINDOW: int = 100
    REWARD_THRESHOLD: float = 0.7
    HYPERPARAM_ADJUST_INTERVAL: int = 100

    # Regularization  (single source of truth — used by AdamW weight_decay)
    WEIGHT_DECAY: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}

    def adjust_learning_rate(self, current_lr: float, performance: float) -> float:
        if performance > self.REWARD_THRESHOLD:
            return min(current_lr * 1.01, self.LR_MAX)
        else:
            return max(current_lr * 0.99, self.LR_MIN)


# ---------------------------------------------------------------------------
# 4. Simulation configuration
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Simulation configuration"""
    EPISODE_LENGTH: int = 100
    MAX_STEPS_PER_EPISODE: int = 50
    MAX_CONCURRENT_FLOWS: int = 20
    INITIAL_FLOWS: int = 15            # flows pre-seeded at env.reset()
    FLOW_ARRIVAL_RATE: float = 0.5
    MIN_FLOW_DURATION: int = 10
    MAX_FLOW_DURATION: int = 100

    TIME_STEP_MS: float = 10.0
    WARMUP_STEPS: int = 100

    SAVE_INTERVAL: int = 50
    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = True
    DEBUG: bool = False

    RENDER: bool = False
    RENDER_INTERVAL: int = 10

    EVALUATION_INTERVAL: int = 10
    EVALUATION_EPISODES: int = 5

    SAVE_CHECKPOINTS: bool = True
    CHECKPOINT_INTERVAL: int = 20

    METRICS_WINDOW: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}


# ---------------------------------------------------------------------------
# 5. GNN feature-dimension contract (NEW)
# ---------------------------------------------------------------------------
@dataclass
class GNNConfig:
    """Single source of truth for GNN tensor shapes."""
    NODE_FEAT_DIM: int = 7      # matches environment._get_node_features
    EDGE_FEAT_DIM: int = 9      # matches environment._get_edge_features
    HIDDEN_DIM: int = 64
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.1
    MODEL_TYPE: str = "simple"  # 'simple' or 'full'


# ---------------------------------------------------------------------------
# Default instances and helpers
# ---------------------------------------------------------------------------
DEFAULT_SATELLITE_CONFIG = SatelliteConfig()
DEFAULT_TRAFFIC_CONFIG = TrafficConfig()
DEFAULT_DQN_CONFIG = DQNConfig()
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_GNN_CONFIG = GNNConfig()


class OptimizedConfigs:
    """Pre-configured scenarios."""

    @staticmethod
    def get_debug_config():
        """Small but **heavily congested** scenario for fast iteration.
        """
        sat = SatelliteConfig()
        sat.NUM_SATELLITES = 24
        sat.NUM_ORBITS = 4
        sat.SATELLITES_PER_ORBIT = 6
        sat.GROUND_STATIONS = 5
        # Smaller links — a single high-demand flow can saturate one
        sat.ISL_BANDWIDTH = 30e6         # 30 Mbps
        sat.SATELLITE_BANDWIDTH = 60e6   # 60 Mbps
        sat.GSL_BANDWIDTH = 100e6        # 100 Mbps
        sat.MAX_ALLOWED_DELAY_MS = 50.0  # tighter QoS

        sim = SimulationConfig()
        sim.MAX_CONCURRENT_FLOWS = 60
        sim.EPISODE_LENGTH = 50
        sim.INITIAL_FLOWS = 50           # heavy initial load
        return sat, TrafficConfig(), DQNConfig(), sim

    @staticmethod
    def get_training_config():
        sat = SatelliteConfig()
        sat.NUM_SATELLITES = 72
        sat.NUM_ORBITS = 3
        sat.SATELLITES_PER_ORBIT = 24
        sat.GROUND_STATIONS = 10
        sim = SimulationConfig()
        sim.MAX_CONCURRENT_FLOWS = 10
        return sat, TrafficConfig(), DQNConfig(), sim

    @staticmethod
    def get_production_config():
        sat = SatelliteConfig()
        dqn = DQNConfig()
        dqn.USE_PER = True
        dqn.USE_DOUBLE_DQN = True
        dqn.USE_DUELING = True
        return sat, TrafficConfig(), dqn, SimulationConfig()


def validate_configs(sat_config: SatelliteConfig,
                     traffic_config: TrafficConfig,
                     dqn_config: DQNConfig,
                     sim_config: SimulationConfig) -> Dict[str, Any]:
    """Return a small validation summary."""
    max_net = sat_config.SATELLITE_BANDWIDTH * sat_config.NUM_SATELLITES
    max_demand = sim_config.MAX_CONCURRENT_FLOWS * max(
        ft['bw_req'] for ft in traffic_config.FLOW_TYPES.values()
    )

    warnings = []
    if max_demand > max_net * 0.8:
        warnings.append(
            f"High network load: demand={max_demand/1e9:.1f} Gbps > 80% of "
            f"capacity {max_net/1e9:.1f} Gbps"
        )
    if sat_config.MAX_ALLOWED_DELAY_MS < 50:
        warnings.append(f"Very strict delay requirement: "
                        f"{sat_config.MAX_ALLOWED_DELAY_MS} ms")

    return {
        'satellite_count': sat_config.NUM_SATELLITES,
        'network_capacity_gbps': max_net / 1e9,
        'max_concurrent_flows': sim_config.MAX_CONCURRENT_FLOWS,
        'episode_length': sim_config.EPISODE_LENGTH,
        'learning_rate': dqn_config.LR,
        'buffer_size': dqn_config.BUFFER_SIZE,
        'state_dim': dqn_config.STATE_DIM,
        'action_dim': dqn_config.ACTION_DIM,
        'warnings': warnings,
        'config_valid': len(warnings) == 0,
    }
