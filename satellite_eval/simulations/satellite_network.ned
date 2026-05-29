// satellite_network.ned
package satellite_eval;

import inet.node.inet.StandardHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.linklayer.ethernet.EthernetLink;

network SatelliteNetwork
{
    parameters:
        int num_orbits = default(4);
        int sats_per_orbit = default(18);
        @display("bgb=800,600");
    submodules:
        // Dynamically create satellites using a loop (OMNeT++ 6.0+ supports loops in NED)
        for orbit=0..num_orbits-1 {
            for pos=0..sats_per_orbit-1 {
                sat[orbit][pos]: StandardHost {
                    @display("p=??"); // position will be set in C++
                    parameters:
                        @sateval(nodeType = "satellite");
                }
            }
        }
        // Ground stations (e.g., 10)
        gs[0..9]: StandardHost {
            @display("p=??");
            parameters:
                @sateval(nodeType = "ground");
        }
        // Controller (ZeroMQ interface)
        controller: SatelliteController {
            @display("p=500,300");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=700,50");
        }
    connections:
        // Links will be created programmatically in C++ due to dynamic topology
}