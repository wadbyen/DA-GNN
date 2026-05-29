// SatelliteController.h
#ifndef SATELLITECONTROLLER_H
#define SATELLITECONTROLLER_H

#include <omnetpp.h>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_map>
#include <random>

using namespace omnetpp;
using json = nlohmann::json;

class SatelliteController : public cSimpleModule
{
  private:
    // ZeroMQ
    zmq::context_t context;
    zmq::socket_t socket;
    cMessage *stepTrigger;
    simtime_t stepDuration;

    // Topology
    std::vector<cModule*> satellites;
    std::vector<cModule*> groundStations;
    std::vector<cGate*> allGates; // for link iteration

    // Flow management
    struct Flow {
        int id;
        int srcIdx;
        int dstIdx;
        std::string type;
        double demand;          // bps
        int duration;           // timesteps
        simtime_t startTime;
        simtime_t deadline;
        double allocatedBW;
        double actualThroughput;
        double endToEndDelay;
        double packetLoss;
        std::vector<int> route;   // indices of nodes
        bool active;
        int priority;
    };
    std::vector<Flow> flows;
    std::unordered_map<int, std::vector<int>> activeRoutes; // flowId -> route indices

    // Link states (matching Python's link_states)
    struct LinkState {
        double utilization;
        double delay;          // seconds
        double loss;
        double availableBW;
        double capacity;
        bool failed;
        double failureProb;
        int queueSize;
        double congestionLevel;
        int packetsDropped;
        int activeFlows;
        double allocatedBW;
    };
    std::unordered_map<std::string, LinkState> linkStates; // key "u-v"

    // Node states
    struct NodeState {
        int queueLength;
        double processingDelay;
        int bufferSize;
        int activeFlows;
        int packetsProcessed;
        int packetsDropped;
    };
    std::unordered_map<int, NodeState> nodeStates;

    // Traffic & QoS config (from config.py)
    std::unordered_map<std::string, json> flowTypes;
    std::vector<std::string> trafficPatterns;
    int timeStep;

    // Random generators
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist01;

    // Helper methods
    void createConstellation();
    void createLinks();
    void initializeStates();
    void updateNetworkDynamics();
    void generateTraffic();
    bool allocateBandwidthOnRoute(Flow &flow, const std::vector<int> &route);
    void releaseBandwidthOnRoute(Flow &flow);
    json calculateRouteMetrics(const std::vector<int> &route);
    void collectState(json &state);
    void applyRouting(const json &action);
    double computeReward();
    void updateFlowMetrics();
    void resetSimulation();

    double calculatePropagationDelay(double distanceKm);
    double calculateIntraOrbitDistance();
    double calculateInterOrbitDistance();
    double calculateSGLDistance(int satIdx, int gsIdx);

  public:
    SatelliteController();
    virtual ~SatelliteController();
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

#endif