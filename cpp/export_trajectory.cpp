#include "rocket_control.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace rocket;

int main() {
    RocketParams rocket;
    ControlParams ctrl;
    
    Simulator sim(rocket, ctrl, 0.001);
    
    // Initial conditions matching Python
    State initial(20.0, 300.0, 2.0 * DEG2RAD, 5.0, -10.0, 0.0);
    
    Simulator::Result result = sim.runLanding(initial, 50.0);
    
    // Export to CSV
    std::ofstream file("cpp_trajectory.csv");
    file << std::setprecision(10);
    file << "t,x,z,theta,xdot,zdot,thetadot,TWR,delta\n";
    
    const State* states = sim.getStateHistory();
    const ControlOutput* controls = sim.getControlHistory();
    size_t n = sim.getHistoryLength();
    
    for (size_t i = 0; i < n; ++i) {
        file << i * 0.001 << ","
             << states[i].x << ","
             << states[i].z << ","
             << states[i].theta << ","
             << states[i].xdot << ","
             << states[i].zdot << ","
             << states[i].thetadot << ","
             << controls[i].TWR << ","
             << controls[i].delta << "\n";
    }
    
    file.close();
    
    std::cout << "Exported " << n << " points to cpp_trajectory.csv\n";
    std::cout << "Landing time: " << result.landing_time << "s\n";
    std::cout << "Final x: " << result.final_x << "m\n";
    std::cout << "Final vz: " << result.final_vz << "m/s\n";
    
    return 0;
}
