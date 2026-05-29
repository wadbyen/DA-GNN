"""
environment_n.py - Satellite Network Environment with dynamic conditions
"""

import numpy as np
import networkx as nx
import random
import math
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import SatelliteConfig, TrafficConfig, SimulationConfig

logger = logging.getLogger(__name__)


def _canonical_key(u, v) -> Tuple:
    """Single canonical representation for an undirected link."""
    return tuple(sorted((str(u), str(v))))


class SatelliteNetworkEnvironment:
    """LEO Satellite Network Environment with dynamic conditions"""

    # Public feature dimensions (consumed by GNN / DQN preprocessing)
    NODE_FEAT_DIM = 7
    EDGE_FEAT_DIM = 9

    def __init__(self,
                 sat_config: SatelliteConfig,
                 traffic_config: TrafficConfig,
                 sim_config: Optional[SimulationConfig] = None):
        self.config = sat_config
        self.traffic_config = traffic_config
        self.sim_config = sim_config or SimulationConfig()

        self.graph = self._create_walker_delta_constellation()
        self.link_states: Dict[Tuple, Dict[str, Any]] = {}
        self.node_states: Dict[Any, Dict[str, Any]] = {}
        self.flows: List[Dict[str, Any]] = []
        self.active_flows: List[Dict[str, Any]] = []
        self.time_step = 0
        self.max_flows = self.sim_config.MAX_CONCURRENT_FLOWS * 5
        self.episode_length = self.sim_config.EPISODE_LENGTH

        # Pre-compute node index map for fast lookups
        self._node_idx_map = {n: i for i, n in enumerate(self.graph.nodes())}

        self._initialize_states()

        # Performance metrics
        self.metrics_history = {
            'delays': [], 'losses': [], 'throughputs': [],
            'qos_violations': [], 'link_failures': [],
        }

        logger.info(f"Environment initialized with {self.graph.number_of_nodes()} "
                    f"nodes and {self.graph.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Topology construction
    # ------------------------------------------------------------------
    def _create_walker_delta_constellation(self) -> nx.Graph:
        G = nx.Graph()
        sats_per_orbit = self.config.SATELLITES_PER_ORBIT

        # Satellites
        for orbit in range(self.config.NUM_ORBITS):
            for pos in range(sats_per_orbit):
                sat_id = orbit * sats_per_orbit + pos
                G.add_node(sat_id, type='satellite', orbit=orbit,
                           position=pos, altitude=self.config.ALTITUDE)

        # Ground stations
        for gs_id in range(self.config.GROUND_STATIONS):
            G.add_node(f"GS_{gs_id}", type='ground_station', id=gs_id)

        self._create_isl_connections(G)
        self._create_sgl_connections(G)
        return G

    def _create_isl_connections(self, G: nx.Graph):
        sats_per_orbit = self.config.SATELLITES_PER_ORBIT
        for orbit in range(self.config.NUM_ORBITS):
            for pos in range(sats_per_orbit):
                sat_id = orbit * sats_per_orbit + pos

                # Intra-orbit (forward)
                nxt = orbit * sats_per_orbit + ((pos + 1) % sats_per_orbit)
                if not G.has_edge(sat_id, nxt):
                    G.add_edge(sat_id, nxt,
                               type='intra_orbit',
                               distance=self._calculate_intra_orbit_distance(),
                               bandwidth=self.config.ISL_BANDWIDTH)

                # Inter-orbit
                if orbit < self.config.NUM_ORBITS - 1:
                    n_orb = (orbit + 1) * sats_per_orbit + pos
                    if not G.has_edge(sat_id, n_orb):
                        G.add_edge(sat_id, n_orb,
                                   type='inter_orbit',
                                   distance=self._calculate_inter_orbit_distance(),
                                   bandwidth=self.config.ISL_BANDWIDTH)

                # Cross-seam (only near polar orbits)
                if self.config.ORBITAL_INCLINATION > 80 and orbit == 0 \
                        and pos < sats_per_orbit // 2:
                    opp_orb = self.config.NUM_ORBITS - 1
                    opp_pos = (pos + sats_per_orbit // 2) % sats_per_orbit
                    opp_sat = opp_orb * sats_per_orbit + opp_pos
                    if not G.has_edge(sat_id, opp_sat):
                        G.add_edge(sat_id, opp_sat,
                                   type='cross_seam',
                                   distance=self._calculate_cross_seam_distance(),
                                   bandwidth=self.config.ISL_BANDWIDTH)

    def _create_sgl_connections(self, G: nx.Graph):
        sats_per_orbit = self.config.SATELLITES_PER_ORBIT
        for gs_id in range(self.config.GROUND_STATIONS):
            gs_node = f"GS_{gs_id}"
            orbit = random.randint(0, self.config.NUM_ORBITS - 1)
            positions = random.sample(range(sats_per_orbit),
                                      min(4, sats_per_orbit))
            for pos in positions:
                sat_id = orbit * sats_per_orbit + pos
                G.add_edge(gs_node, sat_id,
                           type='sgl',
                           distance=self._calculate_sgl_distance(sat_id, gs_id),
                           bandwidth=self.config.GSL_BANDWIDTH)

    def _calculate_intra_orbit_distance(self) -> float:
        r = self.config.EARTH_RADIUS + self.config.ALTITUDE
        ang = 360 / self.config.SATELLITES_PER_ORBIT
        return 2 * r * math.sin(math.radians(ang / 2))

    def _calculate_inter_orbit_distance(self) -> float:
        spacing = 360 / self.config.NUM_ORBITS
        r = self.config.EARTH_RADIUS + self.config.ALTITUDE
        return 2 * r * math.sin(math.radians(spacing / 2))

    def _calculate_cross_seam_distance(self) -> float:
        r = self.config.EARTH_RADIUS + self.config.ALTITUDE
        return 2 * r * math.sin(math.radians(90))

    def _calculate_sgl_distance(self, sat_id: int, gs_id: int) -> float:
        min_elev = 20  # degrees
        er = self.config.EARTH_RADIUS
        h = self.config.ALTITUDE
        max_d = math.sqrt((er + h)**2 -
                          (er * math.cos(math.radians(min_elev)))**2) \
            - er * math.sin(math.radians(min_elev))
        return random.uniform(h, max_d)

    # ------------------------------------------------------------------
    # State (re)initialization
    # ------------------------------------------------------------------
    def _initialize_states(self):
        self.link_states.clear()
        for u, v, data in self.graph.edges(data=True):
            key = _canonical_key(u, v)
            self.link_states[key] = {
                'utilization': random.uniform(0.05, 0.15),
                'delay': self._calculate_propagation_delay_s(data['distance']),
                'loss': random.uniform(0.0005, 0.005),
                'available_bandwidth': data['bandwidth'],
                'capacity': data['bandwidth'],
                'failed': False,
                'failure_prob': 0.001,
                'queue_size': 0,
                'congestion_level': 0.0,
                'packets_dropped': 0,
                'active_flows': 0,
                'allocated_bandwidth': 0.0,
            }
        self.node_states.clear()
        for node in self.graph.nodes():
            self.node_states[node] = {
                'queue_length': 0,
                'processing_delay': random.uniform(0.0005, 0.002),  # 0.5-2 ms
                'buffer_size': 1000,
                'active_flows': 0,
                'packets_processed': 0,
                'packets_dropped': 0,
            }

    def _calculate_propagation_delay_s(self, distance_km: float) -> float:
        """Propagation delay in **seconds**."""
        return distance_km / self.config.PROPAGATION_SPEED

    # ------------------------------------------------------------------
    # Reset / flow generation
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.time_step = 0
        self.flows = []
        self.active_flows = []
        self.last_flow_rewards = {}
        self._initialize_states()
        # Heavier default load — 40 flows instead of 15 — so links can
        # actually saturate and path choice matters.
        n_initial = int(getattr(self.sim_config, 'INITIAL_FLOWS', 40))
        self._generate_initial_flows(num_initial_flows=n_initial)
        for k in self.metrics_history:
            self.metrics_history[k] = []
        logger.info(f"Environment reset, {len(self.flows)} initial flows")
        return self.get_state()

    def _generate_initial_flows(self, num_initial_flows: int = 40):
        """Generate initial flows. The default count is now 40 (was 15)
        """
        nodes = list(self.graph.nodes())
        flow_type_names = list(self.traffic_config.FLOW_TYPES.keys())
        flow_type_weights = [self.traffic_config.FLOW_TYPES[ft]['priority']
                             for ft in flow_type_names]

        for i in range(num_initial_flows):
            src = random.choice(nodes)
            dst = random.choice(nodes)
            while dst == src:
                dst = random.choice(nodes)

            # *** FIX: get a STRING name, not an enum  ***
            flow_type_name = random.choices(flow_type_names,
                                            weights=flow_type_weights)[0]
            self.flows.append(self._make_flow(i, src, dst, flow_type_name))

    def _make_flow(self, flow_id: int, src, dst, flow_type: str) -> Dict[str, Any]:
        """Construct a flow dict given a *string* flow type."""
        qos_req = self.traffic_config.FLOW_TYPES[flow_type]

        # Demand & duration depend on the flow type
        if flow_type == 'voip':
            demand = random.uniform(64e3, 128e3)
            duration = random.randint(60, 300)
        elif flow_type == 'video':
            demand = random.uniform(1e6, 5e6)
            duration = random.randint(300, 600)
        elif flow_type == 'gaming':
            demand = random.uniform(2e6, 10e6)
            duration = random.randint(60, 240)
        elif flow_type == 'control':
            demand = random.uniform(5e3, 50e3)
            duration = random.randint(30, 180)
        elif flow_type == 'web':
            demand = random.uniform(500e3, 5e6)
            duration = random.randint(30, 120)
        elif flow_type == 'file_transfer':
            demand = random.uniform(10e6, 100e6)
            duration = random.randint(100, 300)
        else:  # generic data
            demand = random.uniform(1e6, 20e6)
            duration = random.randint(60, 240)

        return {
            'id': flow_id,
            'src': src,
            'dst': dst,
            'type': flow_type,          # *** STRING ***
            'demand': demand,
            'duration': duration,
            'start_time': self.time_step,
            'deadline': self.time_step + qos_req['delay_req'],
            'qos_requirements': qos_req,
            'metrics': {},
            'route': [],
            'status': 'active',
            'allocated_bandwidth': 0,
            'actual_throughput': 0,
            'end_to_end_delay': 0,
            'packet_loss': 0,
            'priority': qos_req['priority'],
            'source': src,
            'destination': dst,
            'arrival_time': self.time_step,
        }

    # ------------------------------------------------------------------
    # State getters
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            'node_features': self._get_node_features(),
            'edge_features': self._get_edge_features(),
            'edge_index': self._get_edge_index(),
            'traffic_matrix': self._get_traffic_matrix(),
            'link_capacities': self._get_link_capacities(),
            'queue_sizes': self._get_queue_sizes(),
            'active_flows': len([f for f in self.flows
                                 if f['status'] == 'active']),
            'link_utilization': self._get_link_utilizations(),
            'link_utilizations': self._get_link_utilizations(),  # alias
            'link_delays': self._get_link_delays(),
            'delays': self._get_link_delays(),                    # alias
            'link_losses': self._get_link_losses(),
            'time_of_day': self.time_step % 24,
            'graph': self.graph,
        }

    def _get_node_features(self) -> np.ndarray:
        """7 features per node."""
        feats = []
        for node, st in self.node_states.items():
            node_type = 1.0 if 'GS_' in str(node) else 0.0
            feats.append([
                st['queue_length'] / 1000.0,
                st['processing_delay'] * 1000.0,  # ms
                st['active_flows'] / 10.0,
                node_type,
                (self.time_step % 24) / 24.0,
                st['packets_processed'] / 1e6,
                st['packets_dropped'] / 1e4,
            ])
        return np.array(feats, dtype=np.float32)

    def _get_edge_features(self) -> np.ndarray:
        """9 features per edge."""
        feats = []
        for (u, v), st in self.link_states.items():
            # Look up the edge type from the graph (handle either ordering)
            if self.graph.has_edge(u, v):
                edge_data = self.graph[u][v]
            elif self.graph.has_edge(v, u):
                edge_data = self.graph[v][u]
            else:
                edge_data = {}

            etype = edge_data.get('type', 'unknown')
            type_code = {'intra_orbit': 0.0, 'inter_orbit': 1.0,
                         'cross_seam': 2.0, 'sgl': 3.0}.get(etype, 4.0)

            util = st.get('utilization', 0.0)
            delay_ms = st.get('delay', 0.01) * 1000.0
            loss = st.get('loss', 0.001)
            cap = st.get('capacity', self.config.ISL_BANDWIDTH)
            allocated = st.get('allocated_bandwidth', 0.0)
            avail_norm = max(0.0, cap - allocated) / self.config.ISL_BANDWIDTH
            failed = 1.0 if st.get('failed', False) else 0.0
            queue = st.get('queue_size', 0) / 1000.0
            cong = st.get('congestion_level', 0.0)
            drops = st.get('packets_dropped', 0) / 1e4

            feats.append([util, delay_ms, loss, avail_norm, failed,
                          type_code, queue, cong, drops])
        return np.array(feats, dtype=np.float32)

    def _get_edge_index(self) -> np.ndarray:
        """Edge index for GNN: shape [2, num_directed_edges]."""
        idx_map = self._node_idx_map
        src_list, dst_list = [], []
        for u, v in self.graph.edges():
            ui, vi = idx_map[u], idx_map[v]
            src_list.append(ui); dst_list.append(vi)
            src_list.append(vi); dst_list.append(ui)  # symmetric
        if not src_list:
            return np.array([[0], [0]], dtype=np.int64)
        return np.array([src_list, dst_list], dtype=np.int64)

    def _get_traffic_matrix(self) -> np.ndarray:
        n = self.graph.number_of_nodes()
        m = np.zeros((n, n), dtype=np.float32)
        idx = self._node_idx_map
        for flow in self.flows[-50:]:
            if flow['status'] == 'active' \
                    and flow['src'] in idx and flow['dst'] in idx:
                si = idx[flow['src']]
                di = idx[flow['dst']]
                m[si, di] += flow['demand'] / self.config.ISL_BANDWIDTH
        return m

    def _get_link_capacities(self) -> np.ndarray:
        return np.array([[s['capacity'] / self.config.ISL_BANDWIDTH]
                         for s in self.link_states.values()],
                        dtype=np.float32)

    def _get_queue_sizes(self) -> np.ndarray:
        return np.array([[s['queue_size'] / 1000.0]
                         for s in self.link_states.values()],
                        dtype=np.float32)

    def _get_link_utilizations(self) -> np.ndarray:
        return np.array([s['utilization']
                         for s in self.link_states.values()],
                        dtype=np.float32)

    def _get_link_delays(self) -> np.ndarray:
        return np.array([s['delay'] * 1000.0    # to ms
                         for s in self.link_states.values()],
                        dtype=np.float32)

    def _get_link_losses(self) -> np.ndarray:
        return np.array([s['loss']
                         for s in self.link_states.values()],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Dict[int, List]) -> Tuple[Dict, float, bool, Dict]:
        self.time_step += 1
        self._update_network_dynamics()
        reward = self._process_flows_with_action(action)
        self._generate_traffic()

        next_state = self.get_state()
        done = self.time_step >= self.episode_length
        info = self._collect_metrics()

        if self.time_step % 10 == 0:
            logger.debug(f"Step {self.time_step}: r={reward:.3f}, "
                         f"active={info['active_flows']}, "
                         f"avg_delay={info['avg_delay']:.2f}ms")
        return next_state, reward, done, info

    def _update_network_dynamics(self):
        hour = self.time_step % 24
        for (u, v), st in self.link_states.items():
            # Safe defaults
            for k, default in [('failure_prob', 0.001), ('failed', False),
                               ('utilization', 0.0),
                               ('allocated_bandwidth', 0.0),
                               ('capacity', self.config.ISL_BANDWIDTH),
                               ('queue_size', 0), ('congestion_level', 0.0),
                               ('loss', 0.001), ('packets_dropped', 0),
                               ('active_flows', 0), ('delay', 0.01)]:
                st.setdefault(k, default)

            base_fp = st['failure_prob']
            mult = 3.0 if st['utilization'] > 0.8 else \
                   1.5 if st['utilization'] > 0.6 else 1.0
            if 8 <= hour <= 20:
                mult *= 1.2
            fp = min(0.05, base_fp * mult)

            if random.random() < fp:
                st['failed'] = True
                st['allocated_bandwidth'] = 0
                st['active_flows'] = 0
            elif random.random() < 0.05:
                st['failed'] = False

            if not st['failed']:
                st['utilization'] = st['allocated_bandwidth'] / st['capacity']

                target_q = int(st['utilization'] * 1000)
                st['queue_size'] = int(0.8 * st['queue_size'] + 0.2 * target_q)

                # Propagation delay
                edge_data = self.graph[u][v] if self.graph.has_edge(u, v) \
                    else self.graph[v][u] if self.graph.has_edge(v, u) else {}
                prop = self._calculate_propagation_delay_s(
                    edge_data.get('distance', 500))
                # Queueing delay (s): assume 1500B pkts at link rate
                pkt_size_bits = 1500 * 8
                serv_rate = max(st['capacity'], 1e3)
                queue_delay = (st['queue_size'] * pkt_size_bits) / serv_rate
                st['delay'] = prop + queue_delay

                u_ = st['utilization']
                st['congestion_level'] = (
                    1.0 if u_ > 0.9 else
                    0.7 if u_ > 0.7 else
                    0.3 if u_ > 0.5 else 0.0
                )
                # loss curve
                st['loss'] = (min(0.1, u_ ** 3) if st['congestion_level'] > 0.7
                              else max(0.001, u_ ** 2 * 0.05))
            else:
                st['delay'] = 1.0       # 1 s = unusable
                st['congestion_level'] = 1.0

    # ------------------------------------------------------------------
    # Bandwidth allocation
    # ------------------------------------------------------------------
    def _allocate_bandwidth_on_route(self, flow: Dict, route: List) -> Tuple[bool, float]:
        if not route or len(route) < 2:
            return False, 0.0

        demand = flow['demand']
        min_avail = float('inf')

        for i in range(len(route) - 1):
            key = _canonical_key(route[i], route[i + 1])
            if key not in self.link_states:
                return False, 0.0
            st = self.link_states[key]
            if st.get('failed', False):
                return False, 0.0
            avail = max(0.0, st['capacity'] - st.get('allocated_bandwidth', 0.0))
            min_avail = min(min_avail, avail)

        if min_avail <= 0:
            return False, 0.0

        alloc = min(demand, min_avail)
        for i in range(len(route) - 1):
            key = _canonical_key(route[i], route[i + 1])
            st = self.link_states[key]
            st['allocated_bandwidth'] = st.get('allocated_bandwidth', 0.0) + alloc
            st['active_flows'] = st.get('active_flows', 0) + 1
            st['utilization'] = st['allocated_bandwidth'] / st['capacity']
            st['available_bandwidth'] = st['capacity'] - st['allocated_bandwidth']

        flow['allocated_bandwidth'] = alloc
        flow['actual_throughput'] = alloc
        return True, alloc

    def _release_bandwidth_on_route(self, flow: Dict, route: List):
        if not route or len(route) < 2:
            return
        alloc = flow.get('allocated_bandwidth', 0)
        if alloc <= 0:
            return
        for i in range(len(route) - 1):
            key = _canonical_key(route[i], route[i + 1])
            if key in self.link_states:
                st = self.link_states[key]
                st['allocated_bandwidth'] = max(
                    0, st.get('allocated_bandwidth', 0) - alloc)
                st['active_flows'] = max(0, st.get('active_flows', 0) - 1)
                st['utilization'] = st['allocated_bandwidth'] / st['capacity']
                st['available_bandwidth'] = st['capacity'] - st['allocated_bandwidth']

    # ------------------------------------------------------------------
    # Flow processing & reward
    # ------------------------------------------------------------------
    def _process_flows_with_action(self, action: Dict[int, List]) -> float:
        """Allocate routes from `action`, update flow state, and return the
        """
        self.last_flow_rewards: Dict[int, float] = {}
        if not action:
            return 0.0

        total_reward = 0.0
        processed = 0

        # Release old allocations for flows that are being rerouted
        for f in self.active_flows:
            if f['id'] in action:
                self._release_bandwidth_on_route(f, f.get('route', []))
        self.active_flows = []

        for flow_id, route in action.items():
            if flow_id >= len(self.flows):
                continue
            flow = self.flows[flow_id]
            if flow.get('status', 'active') != 'active':
                continue
            if self.time_step - flow['start_time'] > flow.get('duration', 1000):
                flow['status'] = 'completed'
                # Completion bonus, scaled by priority
                bonus = 0.5 * (flow['qos_requirements']['priority'] / 5.0)
                self.last_flow_rewards[flow_id] = bonus
                total_reward += bonus
                processed += 1
                continue

            ok, alloc_bw = self._allocate_bandwidth_on_route(flow, route)
            if not ok or alloc_bw <= 0:
                flow['status'] = 'failed'
                # Stronger negative reward — failed allocations should hurt
                # specifically the flow that chose the bad path
                pen = -0.5
                self.last_flow_rewards[flow_id] = pen
                total_reward += pen
                processed += 1
                continue

            metrics = self._calculate_route_metrics(route)
            flow['allocated_bandwidth'] = alloc_bw
            flow['actual_throughput'] = alloc_bw * (1 - metrics.get('loss', 0))
            flow['end_to_end_delay'] = metrics.get('delay', float('inf'))
            flow['packet_loss'] = metrics.get('loss', 0)
            flow['route'] = route
            flow['last_updated'] = self.time_step
            flow['metrics'] = metrics

            qos = flow['qos_requirements']

            d_ms = metrics['delay'] if np.isfinite(metrics['delay']) else 1000.0

            def _path_cost(p):
                """Static, action-attributable cost of a candidate path
                in current network state. Used for comparing what the
                agent picked against what it could have picked.
                """
                if not p or len(p) < 2:
                    return 100.0
                cost = 0.5 * (len(p) - 2)  # hop cost
                max_u = 0.0
                sum_u = 0.0
                n = max(len(p) - 1, 1)
                for i in range(len(p) - 1):
                    k = _canonical_key(p[i], p[i + 1])
                    st = self.link_states.get(k, {})
                    u = float(st.get('utilization', 0.0))
                    dms = float(st.get('delay', 0.001)) * 1000.0
                    ls = float(st.get('loss', 0.0))
                    cost += dms                       # per-link delay (ms)
                    cost += ls * 100.0                # loss probability
                    sum_u += u
                    if u > max_u:
                        max_u = u
                cost += 3.0 * (max_u ** 2)            # bottleneck penalty
                cost += 0.3 * (sum_u / n)             # average congestion
                return cost

            # Compute the chosen path's cost and the alternatives' costs.
            chosen_cost = _path_cost(route)
            try:
                alt_paths = self.get_k_shortest_paths(flow['src'], flow['dst'], k=5)
            except Exception:
                alt_paths = [route]
            if not alt_paths:
                alt_paths = [route]
            alt_costs = [_path_cost(p) for p in alt_paths]
            mean_alt = float(np.mean(alt_costs))

            COST_SCALE = 10.0
            comparative = (mean_alt - chosen_cost) / COST_SCALE
            comparative = float(np.clip(comparative, -1.0, 1.0))

            # Small absolute terms so the agent also has a sense of
            # overall QoS quality, not only relative ranking
            qos_violated = d_ms > qos['delay_req']
            absolute_term = 0.0
            if qos_violated:
                absolute_term -= 0.5          # clear QoS miss → hurt
            if (d_ms < 0.5 * qos['delay_req']
                    and alloc_bw >= 0.95 * flow['demand']):
                absolute_term += 0.2          # comfortably-met QoS → small bonus

            # Priority scaling — high-priority flows weigh slightly more
            priority_w = 0.6 + 0.4 * (qos['priority'] / 5.0)   # [0.6, 1.0]
            flow_r = priority_w * (comparative + absolute_term)
            flow_r = float(np.clip(flow_r, -2.5, 2.5))

            self.last_flow_rewards[flow_id] = flow_r
            total_reward += flow_r
            processed += 1
            self.active_flows.append(flow)

        return total_reward / max(processed, 1)

    def _calculate_route_metrics(self, route: List) -> Dict[str, float]:
        if not route or len(route) < 2:
            return {'delay': float('inf'), 'loss': 1.0, 'throughput': 0}

        total_delay_s = 0.0
        survival = 1.0  # probability NOT lost

        for i in range(len(route) - 1):
            key = _canonical_key(route[i], route[i + 1])
            if key not in self.link_states:
                return {'delay': float('inf'), 'loss': 1.0, 'throughput': 0}
            st = self.link_states[key]
            if st['failed']:
                return {'delay': float('inf'), 'loss': 1.0, 'throughput': 0}

            # Node processing delay (skip source)
            if i > 0 and route[i] in self.node_states:
                total_delay_s += self.node_states[route[i]]['processing_delay']

            total_delay_s += st['delay']
            survival *= (1.0 - st['loss'])

        # Destination processing delay
        if route[-1] in self.node_states:
            total_delay_s += self.node_states[route[-1]]['processing_delay']

        return {
            'delay': total_delay_s * 1000.0,   # ms
            'loss':  1.0 - survival,
            'throughput': 0,
        }

    # ------------------------------------------------------------------
    # Continual traffic generation
    # ------------------------------------------------------------------
    def _generate_traffic(self):
        self.flows = [f for f in self.flows if f.get('status') == 'active']

        patterns = self.traffic_config.TRAFFIC_PATTERNS
        pattern = patterns[(self.time_step // 100) % len(patterns)]

        if pattern == 'bursty':
            new = random.randint(5, 15) if random.random() < 0.3 else random.randint(0, 3)
        elif pattern == 'diurnal':
            new = random.randint(3, 8) if 8 <= self.time_step % 24 <= 20 else random.randint(1, 3)
        elif pattern == 'spiky':
            new = random.randint(10, 25) if random.random() < 0.1 else random.randint(0, 2)
        else:                       # uniform
            new = random.randint(2, 6)

        if len(self.flows) >= self.max_flows:
            return
        max_new = min(new, self.max_flows - len(self.flows))
        nodes = list(self.graph.nodes())
        flow_type_names = list(self.traffic_config.FLOW_TYPES.keys())
        flow_type_weights = [self.traffic_config.FLOW_TYPES[ft]['priority']
                             for ft in flow_type_names]

        for _ in range(max_new):
            src = random.choice(nodes)
            dst = random.choice(nodes)
            while dst == src:
                dst = random.choice(nodes)
            flow_type_name = random.choices(flow_type_names,
                                            weights=flow_type_weights)[0]
            self.flows.append(self._make_flow(len(self.flows),
                                              src, dst, flow_type_name))

    # ------------------------------------------------------------------
    # Metrics & path finding
    # ------------------------------------------------------------------
    def _collect_metrics(self) -> Dict[str, float]:
        delays, losses, tputs = [], [], []
        qos_v, active = 0, 0
        for f in self.flows:
            if f.get('status') != 'active':
                continue
            active += 1
            d = f.get('end_to_end_delay', float('inf'))
            l = f.get('packet_loss', 0)
            t = f.get('actual_throughput', 0)
            if not np.isinf(d):
                delays.append(d)
            losses.append(l)
            tputs.append(t)

            qos = f['qos_requirements']
            delay_ok = d <= qos['delay_req']
            loss_ok = l <= qos['loss_req']
            if not (delay_ok and loss_ok):
                qos_v += 1

        link_fail = sum(1 for s in self.link_states.values() if s['failed'])

        avg_delay = float(np.mean(delays)) if delays else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_tput = float(np.mean(tputs)) if tputs else 0.0

        self.metrics_history['delays'].append(avg_delay)
        self.metrics_history['losses'].append(avg_loss)
        self.metrics_history['throughputs'].append(avg_tput)
        self.metrics_history['qos_violations'].append(qos_v)
        self.metrics_history['link_failures'].append(link_fail)

        return {
            'avg_delay': avg_delay,
            'avg_loss': avg_loss,
            'avg_throughput': avg_tput,
            'qos_violations': qos_v,
            'qos_violation_rate': qos_v / max(active, 1),
            'active_flows': active,
            'link_failures': link_fail,
            'time_step': self.time_step,
            # Per-flow rewards from the most recent step (populated by
            # _process_flows_with_action). Empty between resets.
            'flow_rewards': dict(getattr(self, 'last_flow_rewards', {})),
        }

    def get_k_shortest_paths(self, src, dst, k: int = 3) -> List[List]:
        """Return up to k *diverse* paths from src to dst.
        """
        try:
            if not self.graph.has_node(src) or not self.graph.has_node(dst):
                return []
            if not nx.has_path(self.graph, src, dst):
                return []

            # Build an initial weighted graph from current link state.
            def _base_weight(u, v):
                key = _canonical_key(u, v)
                st = self.link_states.get(key)
                if st is None:
                    return 1.0
                if st['failed']:
                    return 1e9
                dw = st['delay'] * 1000.0      # ms
                uw = st['utilization'] * 100.0
                lw = st['loss'] * 1000.0
                w = dw + uw + lw
                if st['congestion_level'] > 0.7:
                    w *= 2.0
                return max(w, 1e-3)

            wg = self.graph.copy()
            for u, v in wg.edges():
                wg[u][v]['weight'] = _base_weight(u, v)

            paths: List[List] = []
            used_edge_counts: Dict[tuple, int] = {}
            # Penalty multiplier applied per prior use of an edge.
            # Bumped from 3.0 → 5.0 because at k=5 we need stronger
            # divergence to find 5 genuinely-different paths in this
            # well-connected satellite topology. With 3.0 the 4th and
            # 5th paths often duplicated the 2nd and 3rd.
            REUSE_PENALTY = 5.0

            for _ in range(k):
                try:
                    p = nx.shortest_path(wg, src, dst, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break
                if not p or len(p) < 2:
                    break

                # Skip duplicates.
                if any(p == q for q in paths):
                    # Heavily penalise this exact route's edges and retry once.
                    for i in range(len(p) - 1):
                        e = tuple(sorted((str(p[i]), str(p[i + 1]))))
                        used_edge_counts[e] = used_edge_counts.get(e, 0) + 2
                else:
                    paths.append(p)
                    for i in range(len(p) - 1):
                        e = tuple(sorted((str(p[i]), str(p[i + 1]))))
                        used_edge_counts[e] = used_edge_counts.get(e, 0) + 1

                # Re-weight: edges seen on earlier paths cost more.
                for u, v in wg.edges():
                    base = _base_weight(u, v)
                    key = tuple(sorted((str(u), str(v))))
                    mult = REUSE_PENALTY ** used_edge_counts.get(key, 0)
                    wg[u][v]['weight'] = base * mult

                if len(paths) >= k:
                    break

            # If we still don't have k paths, top up with shortest_simple_paths
            # so the action space stays the expected size.
            if len(paths) < k:
                try:
                    fill_wg = self.graph.copy()
                    for u, v in fill_wg.edges():
                        fill_wg[u][v]['weight'] = _base_weight(u, v)
                    for p in nx.shortest_simple_paths(fill_wg, src, dst, weight='weight'):
                        if p not in paths:
                            paths.append(p)
                        if len(paths) >= k:
                            break
                except Exception:
                    pass

            return paths[:k]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        except Exception as e:
            logger.warning(f"Path finding {src}->{dst} failed: {e}")
            return []

    def get_all_paths(self, src, dst, max_hops: int = 6) -> List[List]:
        try:
            return list(nx.all_simple_paths(self.graph, src, dst,
                                            cutoff=max_hops))[:10]
        except Exception:
            return []

    def get_network_stats(self) -> Dict[str, Any]:
        total_bw = sum(s['capacity'] for s in self.link_states.values())
        avg_util = float(np.mean([s['utilization']
                                  for s in self.link_states.values()]))
        valid_delays = [s['delay'] * 1000.0
                        for s in self.link_states.values()
                        if not np.isinf(s['delay'])]
        avg_delay = float(np.mean(valid_delays)) if valid_delays else 0.0
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_links': self.graph.number_of_edges(),
            'total_bandwidth': total_bw,
            'avg_utilization': avg_util,
            'avg_delay_ms': avg_delay,
            'failed_links': sum(1 for s in self.link_states.values()
                                if s['failed']),
            'active_flows': len([f for f in self.flows
                                 if f['status'] == 'active']),
            'total_qos_violations': (self.metrics_history['qos_violations'][-1]
                                     if self.metrics_history['qos_violations']
                                     else 0),
        }
