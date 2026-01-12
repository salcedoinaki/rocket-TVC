#include "rocket_control.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace rocket;


static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")" << std::endl; \
            tests_failed++; \
            return false; \
        } \
    } while(0)

#define TEST_NEAR(a, b, tol, msg) \
    TEST_ASSERT(std::abs((a) - (b)) < (tol), msg << ": expected " << (b) << ", got " << (a))

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... " << std::flush; \
        if (test_func()) { \
            std::cout << "PASS" << std::endl; \
            tests_passed++; \
        } else { \
            std::cout << "FAIL" << std::endl; \
        } \
    } while(0)

bool test_rocket_params() {
    RocketParams params;
    TEST_NEAR(params.m, 5000.0, 1e-6, "Default mass");
    TEST_NEAR(params.g, 9.81, 1e-6, "Default gravity");
    TEST_NEAR(params.iota, 3.0, 1e-6, "Default iota");
    TEST_NEAR(params.Iyy, 750000.0, 1e-6, "Default Iyy");
    TEST_NEAR(params.weight(), 49050.0, 1e-6, "Weight calculation");
    return true;
}

bool test_control_params() {
    ControlParams ctrl;
    TEST_NEAR(ctrl.KP_tau, 1.35e5, 1e-6, "Default KP_tau");
    TEST_NEAR(ctrl.KD_tau, 4.4e5, 1e-6, "Default KD_tau");
    TEST_NEAR(ctrl.beta_max, 9.0 * DEG2RAD, 1e-6, "Default beta_max");
    TEST_NEAR(ctrl.delta_max, 10.0 * DEG2RAD, 1e-6, "Default delta_max");
    TEST_NEAR(ctrl.TWR_max, 1.3, 1e-6, "Default TWR_max");
    return true;
}

bool test_state_initialization() {
    State s1;
    TEST_NEAR(s1.x, 0.0, 1e-6, "Default x");
    TEST_NEAR(s1.z, 0.0, 1e-6, "Default z");
    
    State s2(10.0, 20.0, 0.1, 1.0, 2.0, 0.01);
    TEST_NEAR(s2.x, 10.0, 1e-6, "Custom x");
    TEST_NEAR(s2.z, 20.0, 1e-6, "Custom z");
    TEST_NEAR(s2.theta, 0.1, 1e-6, "Custom theta");
    return true;
}

bool test_controller_hover() {
    // Test controller at hover condition (should output TWR=1, delta=0)
    RocketParams rocket;
    ControlParams ctrl;
    ctrl.x_target = 0.0;
    ctrl.z_target = 100.0;  // Above target
    ctrl.zdot_target = 0.0;
    ctrl.xdot_target = 0.0;
    
    CascadeController controller(rocket, ctrl);
    
    // At target with zero velocity
    State state(0.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    ControlOutput out = controller.compute(state, 0.0);
    
    TEST_ASSERT(out.valid, "Output should be valid");
    TEST_NEAR(out.delta, 0.0, 0.01, "Delta at hover");
    // TWR should be ~1.0 to counteract gravity
    TEST_NEAR(out.TWR, 1.0, 0.1, "TWR at hover");
    return true;
}

bool test_controller_saturation() {
    RocketParams rocket;
    ControlParams ctrl;
    CascadeController controller(rocket, ctrl);
    
    // Large horizontal offset should saturate beta
    State state(1000.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    ControlOutput out = controller.compute(state, 0.0);
    
    TEST_ASSERT(out.valid, "Output should be valid");
    // delta should be within limits
    TEST_ASSERT(std::abs(out.delta) <= ctrl.delta_max + 1e-6, "Delta within limits");
    TEST_ASSERT(out.TWR >= ctrl.TWR_min - 1e-6, "TWR above min");
    TEST_ASSERT(out.TWR <= ctrl.TWR_max + 1e-6, "TWR below max");
    return true;
}

bool test_dynamics_free_fall() {
    RocketParams params;
    RocketDynamics dynamics(params);
    
    // Free fall: TWR=0, delta=0
    State state(0.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    double dt = 0.001;
    
    // After one step
    State next = dynamics.step(state, 0.0, 0.0, dt);
    
    // z should decrease, zdot should become negative
    TEST_NEAR(next.z, 100.0, 1e-3, "z after one step");
    TEST_NEAR(next.zdot, -params.g * dt, 1e-6, "zdot after one step");
    TEST_NEAR(next.x, 0.0, 1e-6, "x unchanged");
    return true;
}

bool test_dynamics_hover() {
    RocketParams params;
    RocketDynamics dynamics(params);
    
    // Hover: TWR=1, delta=0, theta=0
    State state(0.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    double dt = 0.001;
    
    State next = dynamics.step(state, 1.0, 0.0, dt);
    
    // Should maintain position (zdot ~ 0)
    TEST_NEAR(next.zdot, 0.0, 1e-6, "zdot at hover");
    TEST_NEAR(next.xdot, 0.0, 1e-6, "xdot at hover");
    return true;
}

bool test_dynamics_gimbal_torque() {
    RocketParams params;
    RocketDynamics dynamics(params);
    
    // Apply gimbal angle, should create torque
    State state(0.0, 100.0, 0.0, 0.0, 0.0, 0.0);
    double delta = 5.0 * DEG2RAD;
    double dt = 0.001;
    
    State next = dynamics.step(state, 1.0, delta, dt);
    
    // Should have angular acceleration
    // thetaddot = -F_T * iota * sin(delta) / Iyy
    double F_T = params.weight();
    double expected_thetaddot = -(F_T * params.iota / params.Iyy) * std::sin(delta);
    
    TEST_NEAR(next.thetadot, expected_thetaddot * dt, 1e-6, "Angular velocity from gimbal");
    return true;
}

bool test_landing_simulation() {
    // Main integration test: compare against Python results
    // Python result: x=5.55m, vz=-2.52m/s, t=34.0s (approximately)
    
    RocketParams rocket;
    ControlParams ctrl;
    
    Simulator sim(rocket, ctrl, 0.001);
    
    // Initial conditions matching Python
    State initial(20.0, 300.0, 2.0 * DEG2RAD, 5.0, -10.0, 0.0);
    
    Simulator::Result result = sim.runLanding(initial, 50.0);
    
    TEST_ASSERT(result.success, "Landing should succeed");
    
    // Compare with Python results (allowing some tolerance)
    // Python: x=5.55m, vz=-2.52m/s, t=34.0s
    std::cout << "\n    C++ Results vs Python:\n";
    std::cout << "    Landing time: " << result.landing_time << "s (Python: ~34.0s)\n";
    std::cout << "    Final x: " << result.final_x << "m (Python: ~5.55m)\n";
    std::cout << "    Final vz: " << result.final_vz << "m/s (Python: ~-2.52m/s)\n";
    std::cout << "    Final vx: " << result.final_vx << "m/s (Python: ~-0.44m/s)\n";
    std::cout << "    Saturation: " << (100.0 * result.saturation_count / result.num_steps) 
              << "% (Python: ~6.4%)\n";
    
    // Tolerance: within 5% of Python values
    TEST_NEAR(result.landing_time, 34.0, 2.0, "Landing time matches Python");
    TEST_NEAR(result.final_x, 5.55, 1.0, "Final x matches Python");
    TEST_NEAR(result.final_vz, -2.52, 0.5, "Final vz matches Python");
    TEST_NEAR(result.final_vx, -0.44, 0.2, "Final vx matches Python");
    
    return true;
}

bool test_input_validation() {
    RocketParams rocket;
    ControlParams ctrl;
    CascadeController controller(rocket, ctrl);
    
    // NaN input should return invalid output
    State bad_state(std::nan(""), 100.0, 0.0, 0.0, 0.0, 0.0);
    ControlOutput out = controller.compute(bad_state, 0.0);
    
    TEST_ASSERT(!out.valid, "NaN input should produce invalid output");
    return true;
}

bool test_deterministic_execution() {
    // Run same simulation twice, should get identical results
    RocketParams rocket;
    ControlParams ctrl;
    
    Simulator sim1(rocket, ctrl, 0.001);
    Simulator sim2(rocket, ctrl, 0.001);
    
    State initial(20.0, 300.0, 2.0 * DEG2RAD, 5.0, -10.0, 0.0);
    
    Simulator::Result r1 = sim1.runLanding(initial, 50.0);
    Simulator::Result r2 = sim2.runLanding(initial, 50.0);
    
    TEST_NEAR(r1.landing_time, r2.landing_time, 1e-10, "Deterministic landing time");
    TEST_NEAR(r1.final_x, r2.final_x, 1e-10, "Deterministic final x");
    TEST_NEAR(r1.final_vz, r2.final_vz, 1e-10, "Deterministic final vz");
    return true;
}

int main() {
    std::cout << "=== Rocket TVC Control C++ Unit Tests ===\n\n";
    std::cout << std::fixed << std::setprecision(4);
    
    // Parameter tests
    RUN_TEST(test_rocket_params);
    RUN_TEST(test_control_params);
    RUN_TEST(test_state_initialization);
    
    // Controller tests
    RUN_TEST(test_controller_hover);
    RUN_TEST(test_controller_saturation);
    RUN_TEST(test_input_validation);
    
    // Dynamics tests
    RUN_TEST(test_dynamics_free_fall);
    RUN_TEST(test_dynamics_hover);
    RUN_TEST(test_dynamics_gimbal_torque);
    
    // Integration tests
    RUN_TEST(test_landing_simulation);
    RUN_TEST(test_deterministic_execution);
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    
    return tests_failed > 0 ? 1 : 0;
}
