// SatelliteController.cc – Key parts
#include "SatelliteController.h"
#include "SatelliteLink.h"
#include <cmath>
#include <algorithm>
#include <chrono>

Define_Module(SatelliteController);

SatelliteController::SatelliteController() : socket(context, ZMQ_REP), rng(std::chrono::steady_clock::now().time_since_epoch().count())
{
    stepTrigger = nullptr;
    stepDuration = 1.0; // 1 second per step
    timeStep = 0;
    dist01 = std::uniform_real_distribution<double>(0.0, 1.0);
}

SatelliteController::~SatelliteController()
{
    cancelAndDelete(stepTrigger);
}

void SatelliteController::initialize()
{
    // Load parameters from config (match config.py)
    stepDuration = par("stepDuration").doubleValue();
    int port = par("zmqPort").intValue();
    socket.bind(("tcp://*:" + std::to_string(port)).c_str());
    stepTrigger = new cMessage("stepTrigger");

    // Initialize flow types (matching TrafficConfig.FLOW_TYPES)
    flowTypes["video"] = {{"delay_req", 150.0}, {"loss_req", 1e-3}, {"bw_req", 5e6}, {"priority", 3}};
    flowTypes["voip"]  = {{"delay_req", 50.0},  {"loss_req", 1e-4}, {"bw_req", 64e3}, {"priority", 4}};
    flowTypes["web"]   = {{"delay_req", 100.0}, {"loss_req", 1e-3}, {"bw_req", 2e6},  {"priority", 2}};
    flowTypes["file"]  = {{"delay_req", 1000.0},{"loss_req", 0.0},  {"bw_req", 10e6}, {"priority", 1}};
    flowTypes["gaming"]={{"delay_req", 30.0},  {"loss_req", 1e-3}, {"bw_req", 5e6},  {"priority", 4}};
    flowTypes["control"]={{ "delay_req", 10.0}, {"loss_req", 1e-5}, {"bw_req", 10e3}, {"priority", 5}};

    trafficPatterns = {"uniform", "bursty", "diurnal", "spiky"};

    createConstellation();
    createLinks();
    initializeStates();

    EV_INFO << "SatelliteController ready on port " << port << endl;
}

void SatelliteController::createConstellation()
{
    // Get parameters from NED or .ini
    int numOrbits = par("numOrbits");
    int satsPerOrbit = par("satsPerOrbit");
    int numGround = par("numGroundStations");

    // Create satellites
    for (int o = 0; o < numOrbits; ++o) {
        for (int p = 0; p < satsPerOrbit; ++p) {
            char name[32];
            sprintf(name, "sat_%d_%d", o, p);
            cModuleType *type = cModuleType::get("inet.node.inet.StandardHost");
            cModule *sat = type->create(name, getSystemModule());
            sat->finalizeParameters();
            sat->buildInside();
            sat->scheduleStart(simTime());
            satellites.push_back(sat);
        }
    }
    // Ground stations
    for (int g = 0; g < numGround; ++g) {
        char name[32];
        sprintf(name, "gs_%d", g);
        cModuleType *type = cModuleType::get("inet.node.inet.StandardHost");
        cModule *gs = type->create(name, getSystemModule());
        gs->finalizeParameters();
        gs->buildInside();
        gs->scheduleStart(simTime());
        groundStations.push_back(gs);
    }
}

void SatelliteController::createLinks()
{
    // Intra-orbit links (same orbit, adjacent positions)
    int satsPerOrbit = par("satsPerOrbit");
    int numOrbits = par("numOrbits");
    double intraDist = calculateIntraOrbitDistance();
    double interDist = calculateInterOrbitDistance();

    // Helper to add link between two modules
    auto addLink = [&](cModule *a, cModule *b, const char *type, double distance) {
        cGate *outGate = a->addGate("ethg", cGate::OUTPUT, true);
        cGate *inGate = b->addGate("ethg", cGate::INPUT, true);
        cChannel *ch = cChannelType::get("SatelliteLink")->create("link");
        ch->par("datarate") = 10e9; // 10 Gbps
        dynamic_cast<SatelliteLink*>(ch)->setDistance(distance);
        outGate->connectTo(inGate, ch);
        ch->initialize();
        // Store link state with key "aIdx-bIdx"
    };

    // Connect satellites in same orbit (ring)
    for (int o = 0; o < numOrbits; ++o) {
        int base = o * satsPerOrbit;
        for (int p = 0; p < satsPerOrbit; ++p) {
            int cur = base + p;
            int next = base + (p+1) % satsPerOrbit;
            addLink(satellites[cur], satellites[next], "intra_orbit", intraDist);
        }
    }
    // Inter-orbit links (adjacent orbits, same position)
    for (int o = 0; o < numOrbits-1; ++o) {
        for (int p = 0; p < satsPerOrbit; ++p) {
            int cur = o*satsPerOrbit + p;
            int next = (o+1)*satsPerOrbit + p;
            addLink(satellites[cur], satellites[next], "inter_orbit", interDist);
        }
    }
    // Ground station links (simplified: connect each GS to nearest 4 satellites)
    for (cModule *gs : groundStations) {
        // For simplicity, pick first 4 satellites
        for (int i = 0; i < 4 && i < (int)satellites.size(); ++i) {
            double dist = calculateSGLDistance(i, 0); // placeholder
            addLink(gs, satellites[i], "sgl", dist);
        }
    }
}

double SatelliteController::calculateIntraOrbitDistance()
{
    double alt = par("altitude").doubleValue(); // km
    double earthR = 6371; // km
    double orbitalR = earthR + alt;
    int satsPerOrbit = par("satsPerOrbit");
    double angleRad = 2 * M_PI / satsPerOrbit;
    return 2 * orbitalR * sin(angleRad / 2);
}

double SatelliteController::calculateInterOrbitDistance()
{
    double alt = par("altitude").doubleValue();
    double earthR = 6371;
    double orbitalR = earthR + alt;
    int numOrbits = par("numOrbits");
    double angleRad = 2 * M_PI / numOrbits;
    return 2 * orbitalR * sin(angleRad / 2);
}

void SatelliteController::initializeStates()
{
    // Initialize link states (same as Python)
    for (auto &kv : linkStates) {
        kv.second.utilization = dist01(rng) * 0.3 + 0.1;
        kv.second.delay = calculatePropagationDelay(500); // dummy
        kv.second.loss = dist01(rng) * 0.009 + 0.001;
        kv.second.availableBW = kv.second.capacity;
        kv.second.failed = false;
        kv.second.failureProb = 0.001;
        kv.second.queueSize = 0;
        kv.second.congestionLevel = 0;
        kv.second.packetsDropped = 0;
        kv.second.activeFlows = 0;
        kv.second.allocatedBW = 0;
    }
    // Node states
    for (int i = 0; i < (int)satellites.size(); ++i) {
        nodeStates[i].queueLength = 0;
        nodeStates[i].processingDelay = dist01(rng) * 0.004 + 0.001;
        nodeStates[i].bufferSize = 1000;
        nodeStates[i].activeFlows = 0;
        nodeStates[i].packetsProcessed = 0;
        nodeStates[i].packetsDropped = 0;
    }
}

void SatelliteController::updateNetworkDynamics()
{
    // Exactly as in environment_n.py: diurnal multiplier, failure probability, queue updates, etc.
    int hour = timeStep % 24;
    double trafficMult = (hour >= 8 && hour <= 20) ? 1.5 : 0.7;

    for (auto &kv : linkStates) {
        auto &st = kv.second;
        // Failure probability based on utilization
        double failMult = 1.0;
        if (st.utilization > 0.8) failMult = 3.0;
        else if (st.utilization > 0.6) failMult = 1.5;
        if (hour >= 8 && hour <= 20) failMult *= 1.2;
        double prob = std::min(0.1, st.failureProb * failMult);
        if (dist01(rng) < prob) {
            st.failed = true;
            st.allocatedBW = 0;
            st.activeFlows = 0;
        } else if (dist01(rng) < 0.02) {
            st.failed = false;
        }
        if (!st.failed) {
            st.utilization = st.allocatedBW / st.capacity;
            int targetQueue = (int)(st.utilization * 1000);
            st.queueSize = (int)(0.8 * st.queueSize + 0.2 * targetQueue);
            // Delay calculation: propagation + queuing
            double propDelay = calculatePropagationDelay(500); // from link distance
            double queuingDelay = st.queueSize * 0.00001;
            st.delay = propDelay + queuingDelay;
            if (st.utilization > 0.9) st.congestionLevel = 1.0;
            else if (st.utilization > 0.7) st.congestionLevel = 0.7;
            else if (st.utilization > 0.5) st.congestionLevel = 0.3;
            else st.congestionLevel = 0.0;
        } else {
            st.delay = INFINITY;
            st.congestionLevel = 1.0;
        }
        // Loss probability
        if (st.congestionLevel > 0.7) st.loss = std::min(0.1, pow(st.utilization, 3));
        else st.loss = std::max(0.001, pow(st.utilization, 2));
        if (st.congestionLevel > 0.5 && dist01(rng) < 0.01)
            st.packetsDropped++;
    }
}

void SatelliteController::generateTraffic()
{
    // Remove completed flows
    flows.erase(std::remove_if(flows.begin(), flows.end(),
        [this](const Flow &f){ return !f.active || simTime() - f.startTime > f.duration; }),
        flows.end());

    // Determine pattern
    int patternIdx = (timeStep / 100) % trafficPatterns.size();
    std::string pattern = trafficPatterns[patternIdx];
    int numNew = 0;
    if (pattern == "bursty") {
        numNew = (dist01(rng) < 0.3) ? (int)(dist01(rng)*10 + 5) : (int)(dist01(rng)*3);
    } else if (pattern == "diurnal") {
        int hour = timeStep % 24;
        numNew = (hour >= 8 && hour <= 20) ? (int)(dist01(rng)*5 + 3) : (int)(dist01(rng)*2 + 1);
    } else {
        numNew = (int)(dist01(rng)*4 + 2);
    }

    // Limit total flows
    int maxFlows = par("maxFlows").intValue();
    int current = flows.size();
    numNew = std::min(numNew, maxFlows - current);
    if (numNew <= 0) return;

    // Select random source/destination (satellites or ground stations)
    std::vector<cModule*> allNodes = satellites;
    allNodes.insert(allNodes.end(), groundStations.begin(), groundStations.end());

    for (int i = 0; i < numNew; ++i) {
        Flow f;
        f.id = flows.size();
        f.srcIdx = dist01(rng) * allNodes.size();
        do {
            f.dstIdx = dist01(rng) * allNodes.size();
        } while (f.dstIdx == f.srcIdx);
        // Select flow type based on priority distribution (like Python)
        std::vector<std::string> types = {"voip","video","web","file","gaming","control"};
        std::vector<double> weights = {0.15, 0.25, 0.3, 0.15, 0.1, 0.05};
        std::discrete_distribution<int> typeDist(weights.begin(), weights.end());
        f.type = types[typeDist(rng)];
        auto &qos = flowTypes[f.type];
        f.priority = qos["priority"];
        f.demand = qos["bw_req"] * (0.8 + 0.4*dist01(rng));
        f.duration = (f.type == "voip") ? (int)(dist01(rng)*240 + 60) :
                     (f.type == "video") ? (int)(dist01(rng)*300 + 300) : (int)(dist01(rng)*200 + 100);
        f.startTime = simTime();
        f.deadline = f.startTime + qos["delay_req"] / 1000.0;
        f.active = true;
        f.allocatedBW = 0;
        flows.push_back(f);
    }
}

bool SatelliteController::allocateBandwidthOnRoute(Flow &flow, const std::vector<int> &route)
{
    // Same logic as Python's _allocate_bandwidth_on_route
    if (route.size() < 2) return false;
    double minAvail = INFINITY;
    for (size_t i = 0; i < route.size()-1; ++i) {
        int u = route[i], v = route[i+1];
        std::string key = std::to_string(u) + "-" + std::to_string(v);
        auto it = linkStates.find(key);
        if (it == linkStates.end() || it->second.failed) return false;
        double avail = it->second.capacity - it->second.allocatedBW;
        if (avail < 0) avail = 0;
        minAvail = std::min(minAvail, avail);
    }
    double alloc = std::min(flow.demand, minAvail);
    if (alloc <= 0) return false;
    for (size_t i = 0; i < route.size()-1; ++i) {
        int u = route[i], v = route[i+1];
        std::string key = std::to_string(u) + "-" + std::to_string(v);
        auto &st = linkStates[key];
        st.allocatedBW += alloc;
        st.activeFlows++;
        st.utilization = st.allocatedBW / st.capacity;
    }
    flow.allocatedBW = alloc;
    flow.actualThroughput = alloc;
    return true;
}

void SatelliteController::applyRouting(const json &action)
{
    // Release previous allocations for flows that are being rerouted
    for (auto &flow : flows) {
        if (flow.active && action.contains(std::to_string(flow.id))) {
            releaseBandwidthOnRoute(flow);
        }
    }
    // Apply new routes
    for (auto &[flowIdStr, routeJson] : action.items()) {
        int flowId = std::stoi(flowIdStr);
        if (flowId >= (int)flows.size()) continue;
        Flow &flow = flows[flowId];
        if (!flow.active) continue;
        std::vector<int> route;
        for (auto &node : routeJson) route.push_back(node.get<int>());
        if (allocateBandwidthOnRoute(flow, route)) {
            flow.route = route;
            // Calculate metrics for this route
            json metrics = calculateRouteMetrics(route);
            flow.endToEndDelay = metrics["delay"];
            flow.packetLoss = metrics["loss"];
            activeRoutes[flowId] = route;
        } else {
            flow.active = false; // failed
        }
    }
}

json SatelliteController::calculateRouteMetrics(const std::vector<int> &route)
{
    double totalDelay = 0;
    double totalLossProb = 1.0;
    for (size_t i = 0; i < route.size()-1; ++i) {
        int u = route[i], v = route[i+1];
        std::string key = std::to_string(u) + "-" + std::to_string(v);
        auto it = linkStates.find(key);
        if (it == linkStates.end() || it->second.failed) {
            return {{"delay", INFINITY}, {"loss", 1.0}};
        }
        auto &st = it->second;
        if (i > 0) totalDelay += nodeStates[u].processingDelay;
        totalDelay += st.delay;
        totalLossProb *= (1.0 - st.loss);
    }
    if (route.size() > 1) totalDelay += nodeStates[route.back()].processingDelay;
    double effectiveLoss = 1.0 - totalLossProb;
    return {{"delay", totalDelay * 1000}, {"loss", effectiveLoss}};
}

double SatelliteController::computeReward()
{
    // Exactly the same as Python's _calculate_reward() using current flow metrics
    double totalReward = 0.0;
    int processed = 0;
    for (auto &flow : flows) {
        if (!flow.active) continue;
        auto &qos = flowTypes[flow.type];
        double delaySat = 0.0;
        if (flow.endToEndDelay < INFINITY) {
            double req = qos["delay_req"];
            delaySat = std::max(0.0, 1.0 - (flow.endToEndDelay / req));
        }
        double throughputSat = std::min(1.0, flow.allocatedBW / std::max(flow.demand, 1e-6));
        double lossSat = std::max(0.0, 1.0 - (flow.packetLoss / std::max((double)qos["loss_req"], 1e-6)));
        double priorityWeight = (double)flow.priority / 4.0;
        double flowReward = (0.4*delaySat + 0.4*throughputSat + 0.2*lossSat) * priorityWeight;
        totalReward += flowReward;
        processed++;
    }
    return (processed > 0) ? (totalReward / processed) : 0.0;
}

void SatelliteController::collectState(json &state)
{
    // Build state matching get_state() in environment_n.py
    std::vector<std::vector<double>> nodeFeat;
    for (int i = 0; i < (int)satellites.size(); ++i) {
        NodeState &ns = nodeStates[i];
        double nodeType = 0.0; // satellite
        nodeFeat.push_back({(double)ns.queueLength, ns.processingDelay*1000,
                           (double)ns.activeFlows, nodeType,
                           (timeStep % 24)/24.0, (double)ns.packetsProcessed,
                           (double)ns.packetsDropped});
    }
    // Edge features (example)
    std::vector<std::vector<double>> edgeFeat;
    for (auto &kv : linkStates) {
        LinkState &st = kv.second;
        double typeCode = 0.0; // depending on link type
        edgeFeat.push_back({st.utilization, st.delay*1000, st.loss,
                            st.availableBW/10e9, st.failed?1.0:0.0,
                            typeCode, (double)st.queueSize/1000.0,
                            st.congestionLevel, (double)st.packetsDropped});
    }
    // Traffic matrix
    int nNodes = satellites.size() + groundStations.size();
    std::vector<std::vector<double>> trafficMat(nNodes, std::vector<double>(nNodes, 0.0));
    for (auto &flow : flows) {
        if (flow.active) {
            trafficMat[flow.srcIdx][flow.dstIdx] += flow.demand / 10e9;
        }
    }
    state["node_features"] = nodeFeat;
    state["edge_features"] = edgeFeat;
    state["traffic_matrix"] = trafficMat;
    state["link_utilizations"] = std::vector<double>();
    state["link_delays"] = std::vector<double>();
    state["link_losses"] = std::vector<double>();
    state["active_flows"] = flows.size();
    state["time_of_day"] = timeStep % 24;
}

void SatelliteController::handleMessage(cMessage *msg)
{
    // Same ZeroMQ handling as before
    if (msg == stepTrigger) {
        json reply;
        collectState(reply["state"]);
        reply["reward"] = computeReward();
        reply["done"] = (timeStep >= par("episodeLength").intValue());
        std::string replyStr = reply.dump();
        socket.send(zmq::buffer(replyStr), zmq::send_flags::none);
        return;
    }
    // ... same as earlier PythonController::handleMessage ...
}