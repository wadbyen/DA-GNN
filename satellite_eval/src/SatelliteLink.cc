// SatelliteLink.cc
#include "SatelliteLink.h"

Define_Channel(SatelliteLink);

void SatelliteLink::initialize()
{
    cDatarateChannel::initialize();
    propagationSpeed = 3e5; // km/s
    // distanceKm must be set before use
}