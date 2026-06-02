# DA-GNN Delay-aware satellite networks traffic optimization 
# Introduction
Delay-aware traffic optimization model for satellite networks that employ Graph Neural Network (GNN)-based attention and GRU with reinforcement learning (DA-GNN). The prediction model combines GNN methods with attention mechanisms and GRU to predict end-to-end delays. The prediction module uses message passing with multi-head attention to capture short- and long-range spatial dependencies among satellites, queues, and flows, while a GRU cell inside each spatio-temporal layer threads a per-node hidden state across consecutive topology snapshots. This explicit temporal recurrence captures satellite orbital dynamics and inter-satellite link handovers. 
The Dueling DQN agent employs these enhanced delay predictions as a state abstraction and selects routes from a k-shortest-paths action space that improves routing performance  
## Files
| File | Status |
|------|--------|
| `config.py` | Configuration classes, DQN, and traffic patterns |
| `environment_n.py` | Satellite Network Environment with dynamic conditions |
| `gnn_model.py` | GNN Delay Predictor with multi-head attention and Spatio-Temporal  |
| `dqn_components.py` | DQN Neural Network Components (PyTorch) |
| `dqn_agent_d.py` |  Delay-Aware Dueling DQN Agent
| `train_dqn_d.py` | Trainer for the Delay-Aware Dueling DQN |
| `utils.py` | Visualization, logging and IO helpers |

## C++ / OMNeT++ files
`SatelliteController.{cc,h}`, `SatelliteLink.{cc,h}`, `satellite_network.ned`,
and `omnetpp.ini` 

## Extention
SPATIO-TEMPORAL EXTENSION (new):
  * SpatioTemporalLayer: a single message-passing block combining
      (a) multi-head graph attention over the CURRENT topology snapshot
          — captures spatial dependencies among nodes, queues, and
            links (short-range via 1-hop attention, long-range via
            stacking multiple layers + global attention pool downstream)
      (b) GRU cell on per-node hidden state propagated across
          consecutive topology snapshots — captures temporal dynamics
          of satellite orbital motion, ISL handovers, congestion
          buildup, and burst dissipation.
  * SpatioTemporalRouteNet: full model stacking N such ST layers,
    with the per-node hidden state of each layer threaded across
    snapshots. Predicts E2E delay using the temporally-aware node
    embeddings and a global attention pool.
  * GNNDelayPredictor now supports ``model_type='spatiotemporal'``
    with automatic hidden-state management across predict_delay calls
    and reset_temporal_state() for episode boundaries.
## Installation
python=3.9 torch==2.5.1 tsai==0.3.0 numpy==1.25.2 torch_geometric==2.3.1

## Run traning model
python train_dqn_d.py --episodes 500


