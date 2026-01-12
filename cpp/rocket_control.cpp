#include "rocket_control.hpp"

namespace rocket {

CascadeController::CascadeController(const RocketParams& rocket, const ControlParams& ctrl)
    : rocket_(rocket), ctrl_(ctrl) {}

double CascadeController::saturate(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

ControlOutput CascadeController::compute(const State& state, double delta_prev) const {
    // Input validation
    if (!std::isfinite(state.x) || !std::isfinite(state.z) || 
        !std::isfinite(state.theta) || !std::isfinite(state.xdot) ||
        !std::isfinite(state.zdot) || !std::isfinite(state.thetadot)) {
        return ControlOutput(1.0, 0.0, false, false);
    }
    
    const double W = rocket_.weight();
    
    // Outer loop
    double e_x = ctrl_.x_target - state.x;
    double e_xdot = ctrl_.xdot_target - state.xdot;
    double xddot_des = ctrl_.KP_x * e_x + ctrl_.KD_x * e_xdot;
    
    double e_z = ctrl_.z_target - state.z;
    double e_zdot = ctrl_.zdot_target - state.zdot;
    double zddot_des = ctrl_.KP_z * e_z + ctrl_.KD_z * e_zdot;
    
    // Thrust magnitude and direction
    double accel_mag = std::sqrt(xddot_des * xddot_des + 
                                 (zddot_des + rocket_.g) * (zddot_des + rocket_.g));
    double F_T = rocket_.m * accel_mag;
    
    // Compute and saturate TWR
    double TWR = F_T / W;
    TWR = saturate(TWR, ctrl_.TWR_min, ctrl_.TWR_max);
    F_T = TWR * W;
    
    // Thrust direction
    double beta_des = std::atan2(xddot_des, zddot_des + rocket_.g);
    beta_des = saturate(beta_des, -ctrl_.beta_max, ctrl_.beta_max);
    
    // Inner loop
    double theta_des = beta_des - delta_prev;
    double e_theta = theta_des - state.theta;
    double tau_des = ctrl_.KP_tau * e_theta - ctrl_.KD_tau * state.thetadot;
    
    // Maximum available torque
    double tau_max = rocket_.iota * F_T * std::sin(ctrl_.delta_max);
    
    // Check saturation
    bool is_saturated = std::abs(tau_des) > tau_max;
    
    // Clip torque
    double tau_clipped = saturate(tau_des, -tau_max, tau_max);
    
    // Compute delta from torque: tau = -F_T * iota * sin(delta)
    double delta = 0.0;
    if (F_T > 1e-6) {
        double arg = tau_clipped / (rocket_.iota * F_T);
        arg = saturate(arg, -1.0, 1.0);
        delta = -std::asin(arg);
    }
    
    return ControlOutput(TWR, delta, true, is_saturated);
}

RocketDynamics::RocketDynamics(const RocketParams& params)
    : params_(params) {}

void RocketDynamics::computeAccelerations(const State& state, double TWR, double delta,
                                           double& xddot, double& zddot, double& thetaddot) const {
    double F_T = TWR * params_.weight();
    double beta = state.theta + delta;
    
    xddot = (F_T / params_.m) * std::sin(beta);
    zddot = (F_T / params_.m) * std::cos(beta) - params_.g;
    thetaddot = -(F_T * params_.iota / params_.Iyy) * std::sin(delta);
}

State RocketDynamics::step(const State& state, double TWR, double delta, double dt) const {
    double xddot, zddot, thetaddot;
    computeAccelerations(state, TWR, delta, xddot, zddot, thetaddot);
    
    State next;
    // Forward Euler integration
    next.xdot = state.xdot + xddot * dt;
    next.zdot = state.zdot + zddot * dt;
    next.thetadot = state.thetadot + thetaddot * dt;
    
    next.x = state.x + state.xdot * dt;
    next.z = state.z + state.zdot * dt;
    next.theta = state.theta + state.thetadot * dt;
    
    return next;
}

Simulator::Simulator(const RocketParams& rocket, const ControlParams& ctrl, double dt)
    : dynamics_(rocket), controller_(rocket, ctrl), dt_(dt), history_length_(0) {
    state_history_ = new State[MAX_STEPS];
    control_history_ = new ControlOutput[MAX_STEPS];
}

Simulator::~Simulator() {
    delete[] state_history_;
    delete[] control_history_;
}

Simulator::Result Simulator::runLanding(const State& initial_state, double max_time) {
    Result result = {};
    result.success = false;
    
    State state = initial_state;
    size_t saturation_count = 0;
    double max_theta_error = 0.0;
    double delta_prev = 0.0;
    
    history_length_ = 0;
    size_t max_steps = static_cast<size_t>(max_time / dt_);
    if (max_steps > MAX_STEPS) max_steps = MAX_STEPS;
    
    for (size_t i = 0; i < max_steps; ++i) {
        // Store history
        if (history_length_ < MAX_STEPS) {
            state_history_[history_length_] = state;
        }
        
        // Compute control with previous delta
        ControlOutput ctrl = controller_.compute(state, delta_prev);
        
        if (history_length_ < MAX_STEPS) {
            control_history_[history_length_] = ctrl;
            history_length_++;
        }
        
        // Track saturation using the saturated flag
        if (ctrl.saturated) {
            saturation_count++;
        }
        
        // Track theta error (theta_des = beta - delta_prev, but we approximate)
        double e_theta = std::abs(state.theta);  // Simplified tracking
        if (e_theta > max_theta_error) {
            max_theta_error = e_theta;
        }
        
        // Update delta_prev for next iteration
        delta_prev = ctrl.delta;
        
        // Step dynamics
        state = dynamics_.step(state, ctrl.TWR, ctrl.delta, dt_);
        
        // Check landing
        if (state.z <= 0.0) {
            result.success = true;
            result.num_steps = i + 1;
            result.saturation_count = saturation_count;
            result.max_theta_error = max_theta_error;
            result.final_x = state.x;
            result.final_z = state.z;
            result.final_vx = state.xdot;
            result.final_vz = state.zdot;
            result.landing_time = (i + 1) * dt_;
            return result;
        }
    }
    
    // Timeout
    result.num_steps = max_steps;
    result.saturation_count = saturation_count;
    result.max_theta_error = max_theta_error;
    result.final_x = state.x;
    result.final_z = state.z;
    result.final_vx = state.xdot;
    result.final_vz = state.zdot;
    result.landing_time = max_time;
    return result;
}

} // namespace rocket
