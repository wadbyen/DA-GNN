// SatelliteLink.h
#ifndef SATELLITELINK_H
#define SATELLITELINK_H

#include <omnetpp.h>

using namespace omnetpp;

class SatelliteLink : public cDatarateChannel
{
  private:
    double distanceKm;      // distance in km
    double propagationSpeed; // km/s (speed of light)
  public:
    SatelliteLink() : cDatarateChannel() {}
    virtual void initialize() override;
    virtual double getDelay() const override { return distanceKm / propagationSpeed; }
    void setDistance(double km) { distanceKm = km; }
};

#endif