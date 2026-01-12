#ifndef ROCKET_CONTROL_HPP
#define ROCKET_CONTROL_HPP

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rocket {

constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;

struct RocketParams {
    double m;
    double g;
    double iota;
    double Iyy;
    
    constexpr RocketParams(double m_ = 5000.0, double g_ = 9.81, 
                           double iota_ = 3.0, double Iyy_ = 750000.0)
        : m(m_), g(g_), iota(iota_), Iyy(Iyy_) {}
    
    double weight() const { return m * g; }
};

struct ControlParams {
    double z_target;
    double zdot_target;
    double x_target;
    double xdot_target;
    
    double KP_tau;
    double KD_tau;
    double KP_x;
    double KD_x;
    double KP_z;
    double KD_z;
    
    double beta_max;
    double delta_max;
    double TWR_min;
    double TWR_max;
    
    constexpr ControlParams()
        : z_target(0.0), zdot_target(-2.0), x_target(0.0), xdot_target(0.0),
          KP_tau(1.35e5), KD_tau(4.4e5),
          KP_x(0.014), KD_x(0.23), KP_z(0.020), KD_z(0.35),
          beta_max(9.0 * DEG2RAD), delta_max(10.0 * DEG2RAD),
          TWR_min(0.0), TWR_max(1.3) {}
};

struct State {
    double x;
    double z;
    double theta;
    double xdot;
    double zdot;
    double thetadot;
    
    constexpr State(double x_ = 0.0, double z_ = 0.0, double theta_ = 0.0,
                    double xdot_ = 0.0, double zdot_ = 0.0, double thetadot_ = 0.0)
        : x(x_), z(z_), theta(theta_), xdot(xdot_), zdot(zdot_), thetadot(thetadot_) {}
};

struct ControlOutput {
    double TWR;
    double delta;
    bool valid;
    bool saturated;
    
    constexpr ControlOutput() : TWR(1.0), delta(0.0), valid(false), saturated(false) {}
    constexpr ControlOutput(double twr, double d, bool v = true, bool sat = false) 
        : TWR(twr), delta(d), valid(v), saturated(sat) {}
};

class CascadeController {
public:
    CascadeController(const RocketParams& rocket, const ControlParams& ctrl);
    ControlOutput compute(const State& state, double delta_prev) const;
    
    const RocketParams& getRocketParams() const { return rocket_; }
    const ControlParams& getControlParams() const { return ctrl_; }
    void setControlParams(const ControlParams& ctrl) { ctrl_ = ctrl; }
    
private:
    RocketParams rocket_;
    ControlParams ctrl_;
    
    static double saturate(double value, double min_val, double max_val);
};

class RocketDynamics {
public:
    explicit RocketDynamics(const RocketParams& params);
    
    State step(const State& state, double TWR, double delta, double dt) const;
    
    void computeAccelerations(const State& state, double TWR, double delta,
                              double& xddot, double& zddot, double& thetaddot) const;
    
    const RocketParams& getParams() const { return params_; }
    
private:
    RocketParams params_;
};

class Simulator {
public:
    static constexpr size_t MAX_STEPS = 50000;
    
    Simulator(const RocketParams& rocket, const ControlParams& ctrl, double dt = 0.001);
    ~Simulator();
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;
    
    struct Result {
        size_t num_steps;
        size_t saturation_count;
        double max_theta_error;
        double final_x;
        double final_z;
        double final_vx;
        double final_vz;
        double landing_time;
        bool success;
    };
    
    Result runLanding(const State& initial_state, double max_time = 100.0);
    
    const State* getStateHistory() const { return state_history_; }
    const ControlOutput* getControlHistory() const { return control_history_; }
    size_t getHistoryLength() const { return history_length_; }
    
private:
    RocketDynamics dynamics_;
    CascadeController controller_;
    double dt_;
    State* state_history_;
    ControlOutput* control_history_;
    size_t history_length_;
};

} // namespace rocket

#endif // ROCKET_CONTROL_HPP
