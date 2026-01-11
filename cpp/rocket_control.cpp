#include "rocket_control.hpp"

namespace rocket {

// ============================================================================
// CascadeController Implementation
// ============================================================================

CascadeController::CascadeController(const RocketParams& rocket, const ControlParams& ctrl)
    : rocket_(rocket), ctrl_(ctrl) {}

double CascadeController::saturate(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

ControlOutput CascadeController::compute(const State& state) const {
    // Input validation
    if (!std::isfinite(state.x) || !std::isfinite(state.z) || 
        !std::isfinite(state.theta) || !std::isfinite(state.xdot) ||
        !std::isfinite(state.zdot) || !std::isfinite(state.thetadot)) {
        return ControlOutput(1.0, 0.0, false);
    }
    
    const double W = rocket_.weight();
    
    // ========== OUTER LOOP: Position -> Desired accelerations ==========
    double e_x = ctrl_.x_target - state.x;
    double e_xdot = ctrl_.xdot_target - state.xdot;
    double xddot_des = ctrl_.KP_x * e_x + ctrl_.KD_x * e_xdot;
    
    double e_z = ctrl_.z_target - state.z;
    double e_zdot = ctrl_.zdot_target - state.zdot;
    double zddot_des = ctrl_.KP_z * e_z + ctrl_.KD_z * e_zdot;
    
    // ========== Convert to thrust vector ==========
    // F_x = m * xddot_des = F_T * sin(beta)
    // F_z = m * (zddot_des + g) = F_T * cos(beta)
    double F_x = rocket_.m * xddot_des;
    double F_z = rocket_.m * (zddot_des + rocket_.g);
    
    double F_T = std::sqrt(F_x * F_x + F_z * F_z);
    double beta = std::atan2(F_x, F_z);
    
    // Saturate beta
    beta = saturate(beta, -ctrl_.beta_max, ctrl_.beta_max);
    
    // Recompute F_T with saturated beta to maintain vertical component
    if (std::cos(beta) > 1e-6) {
        F_T = F_z / std::cos(beta);
    }
    
    // Compute and saturate TWR
    double TWR = F_T / W;
    TWR = saturate(TWR, ctrl_.TWR_min, ctrl_.TWR_max);
    F_T = TWR * W;
    
    // ========== INNER LOOP: Attitude control ==========
    // Desired attitude tracks thrust vector direction
    double theta_des = beta;
    double e_theta = theta_des - state.theta;
    double e_thetadot = 0.0 - state.thetadot;
    
    // Desired torque
    double tau_des = ctrl_.KP_tau * e_theta + ctrl_.KD_tau * e_thetadot;
    
    // tau = -F_T * iota * sin(delta) => delta = -asin(tau / (F_T * iota))
    double max_torque = F_T * rocket_.iota;
    double delta = 0.0;
    
    if (max_torque > 1e-6) {
        double sin_delta = -tau_des / max_torque;
        sin_delta = saturate(sin_delta, -1.0, 1.0);
        delta = std::asin(sin_delta);
    }
    
    // Saturate gimbal angle
    delta = saturate(delta, -ctrl_.delta_max, ctrl_.delta_max);
    
    return ControlOutput(TWR, delta, true);
}

// ============================================================================
// RocketDynamics Implementation
// ============================================================================

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

// ============================================================================
// Simulator Implementation
// ============================================================================

Simulator::Simulator(const RocketParams& rocket, const ControlParams& ctrl, double dt)
    : dynamics_(rocket), controller_(rocket, ctrl), dt_(dt), history_length_(0) {}

Simulator::Result Simulator::runLanding(const State& initial_state, double max_time) {
    Result result = {};
    result.success = false;
    
    State state = initial_state;
    size_t saturation_count = 0;
    double max_theta_error = 0.0;
    const double delta_max = controller_.getControlParams().delta_max;
    
    history_length_ = 0;
    size_t max_steps = static_cast<size_t>(max_time / dt_);
    if (max_steps > MAX_STEPS) max_steps = MAX_STEPS;
    
    for (size_t i = 0; i < max_steps; ++i) {
        // Store history
        if (history_length_ < MAX_STEPS) {
            state_history_[history_length_] = state;
        }
        
        // Compute control
        ControlOutput ctrl = controller_.compute(state);
        
        if (history_length_ < MAX_STEPS) {
            control_history_[history_length_] = ctrl;
            history_length_++;
        }
        
        // Track saturation
        if (std::abs(ctrl.delta) >= delta_max * 0.99) {
            saturation_count++;
        }
        
        // Track theta error
        double theta_des = std::atan2(
            std::sin(state.theta + ctrl.delta), 
            std::cos(state.theta + ctrl.delta)
        );
        double e_theta = std::abs(theta_des - state.theta);
        if (e_theta > max_theta_error) {
            max_theta_error = e_theta;
        }
        
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
