import numpy as np
import matplotlib.pyplot as plt
from main import Params, dynamics, simulate


class ControlParams:
    # Targets
    z_target = 0.0
    zdot_target = -2.0
    x_target = 0.0
    xdot_target = 0.0
    
    # ============== GAIN CALCULATION ==============
    # 
    # INNER LOOP: ωn = 0.42 rad/s, ζ = 0.7
    #   KP_tau = Iyy × ωn² = 750000 × 0.18 = 135000
    #   KD_tau = 2×ζ×ωn×Iyy = 2×0.7×0.42×750000 = 440000
    #
    KP_tau = 1.35e5
    KD_tau = 4.4e5
    
    # OUTER LOOP: tuned for controlled descent
    #
    KP_x = 0.014
    KD_x = 0.23
    KP_z = 0.020
    KD_z = 0.35
    
    # Limits
    beta_max = np.deg2rad(9)
    delta_max = np.deg2rad(10)
    TWR_min = 0.0
    TWR_max = 1.3


def cascade_controller(t, q, p, cp, delta_prev):
    x, z, theta, xdot, zdot, thetadot = q
    
    # --- Outer loop: position/velocity -> desired acceleration ---
    e_z = cp.z_target - z
    e_zdot = cp.zdot_target - zdot
    e_x = cp.x_target - x
    e_xdot = cp.xdot_target - xdot
    
    vdot_x_des = cp.KP_x * e_x + cp.KD_x * e_xdot
    vdot_z_des = cp.KP_z * e_z + cp.KD_z * e_zdot
    
    # Thrust magnitude
    accel_mag = np.sqrt(vdot_x_des**2 + (vdot_z_des + p.g)**2)
    F_T = p.m * accel_mag
    
    # Clip TWR
    TWR = F_T / (p.m * p.g)
    TWR = np.clip(TWR, cp.TWR_min, cp.TWR_max)
    F_T = TWR * p.m * p.g
    
    # Thrust direction
    beta_des = np.arctan2(vdot_x_des, vdot_z_des + p.g)
    beta_des = np.clip(beta_des, -cp.beta_max, cp.beta_max)
    
    # --- Inner loop: attitude -> gimbal angle ---
    theta_des = beta_des - delta_prev
    
    e_theta = theta_des - theta
    tau_des = cp.KP_tau * e_theta - cp.KD_tau * thetadot
    
    # Compute delta from torque
    tau_max = p.iota * F_T * np.sin(cp.delta_max)
    
    # Check if we're saturating
    is_saturated = abs(tau_des) > tau_max
    
    tau_clipped = np.clip(tau_des, -tau_max, tau_max)
    
    arg = tau_clipped / (p.iota * F_T) if F_T > 0 else 0
    delta = -np.arcsin(arg)
    
    return F_T, delta, is_saturated, e_theta


def simulate_controlled(q0, t0, tf, dt, p, cp):
    n_steps = int(np.ceil((tf - t0) / dt)) + 1
    t_hist = np.zeros(n_steps)
    q_hist = np.zeros((n_steps, 6))
    F_hist = np.zeros(n_steps)
    delta_hist = np.zeros(n_steps)
    sat_count = 0
    e_theta_hist = []

    q = np.array(q0, dtype=float)
    t = t0
    delta_prev = 0.0

    for k in range(n_steps):
        t_hist[k] = t
        q_hist[k] = q
        
        F_T, delta, is_sat, e_theta = cascade_controller(t, q, p, cp, delta_prev)
        F_hist[k] = F_T
        delta_hist[k] = delta
        delta_prev = delta
        if is_sat:
            sat_count += 1
        e_theta_hist.append(e_theta)

        if k > 0:
            z, zdot = q[1], q[4]
            if z <= 0.0 and zdot <= 0.0:
                print(f"Saturation: {sat_count}/{k+1} steps ({100*sat_count/(k+1):.1f}%)")
                print(f"Max |e_theta|: {np.rad2deg(max(abs(np.array(e_theta_hist)))):.1f} deg")
                return t_hist[:k+1], q_hist[:k+1], F_hist[:k+1], delta_hist[:k+1]

        if t >= tf:
            print(f"Saturation: {sat_count}/{k+1} steps ({100*sat_count/(k+1):.1f}%)")
            print(f"Max |e_theta|: {np.rad2deg(max(abs(np.array(e_theta_hist)))):.1f} deg")
            return t_hist[:k+1], q_hist[:k+1], F_hist[:k+1], delta_hist[:k+1]

        q = q + dt * dynamics(t, q, F_T, delta, p)
        t += dt

    return t_hist, q_hist, F_hist, delta_hist


def run_landing():
    p = Params()
    cp = ControlParams()
    
    # Initial conditions: 300m up, small drift
    q0 = [20.0, 300.0, np.deg2rad(2), 5.0, -10.0, 0.0]
    
    t, qh, Fh, dh = simulate_controlled(q0, 0.0, 80.0, 0.001, p, cp)
    
    print(f"Landed at t={t[-1]:.2f}s")
    print(f"Final position: x={qh[-1,0]:.2f}m, z={qh[-1,1]:.2f}m")
    print(f"Final velocity: vx={qh[-1,3]:.2f}m/s, vz={qh[-1,4]:.2f}m/s")
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    
    axs[0,0].plot(t, qh[:,1])
    axs[0,0].axhline(0, color='r', linestyle='--')
    axs[0,0].set_xlabel('t [s]')
    axs[0,0].set_ylabel('z [m]')
    axs[0,0].set_title('Altitude')
    axs[0,0].grid(True, alpha=0.3)
    
    axs[0,1].plot(t, qh[:,0])
    axs[0,1].axhline(0, color='r', linestyle='--')
    axs[0,1].set_xlabel('t [s]')
    axs[0,1].set_ylabel('x [m]')
    axs[0,1].set_title('Horizontal Position')
    axs[0,1].grid(True, alpha=0.3)
    
    axs[1,0].plot(t, qh[:,4])
    axs[1,0].axhline(cp.zdot_target, color='r', linestyle='--', label='target')
    axs[1,0].set_xlabel('t [s]')
    axs[1,0].set_ylabel('vz [m/s]')
    axs[1,0].set_title('Vertical Velocity')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    axs[1,1].plot(t, qh[:,3])
    axs[1,1].axhline(0, color='r', linestyle='--')
    axs[1,1].set_xlabel('t [s]')
    axs[1,1].set_ylabel('vx [m/s]')
    axs[1,1].set_title('Horizontal Velocity')
    axs[1,1].grid(True, alpha=0.3)
    
    axs[2,0].plot(t, np.rad2deg(qh[:,2]))
    axs[2,0].set_xlabel('t [s]')
    axs[2,0].set_ylabel('theta [deg]')
    axs[2,0].set_title('Pitch Angle')
    axs[2,0].grid(True, alpha=0.3)
    
    axs[2,1].plot(t, np.rad2deg(dh))
    axs[2,1].set_xlabel('t [s]')
    axs[2,1].set_ylabel('delta [deg]')
    axs[2,1].set_title('Gimbal Angle')
    axs[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_landing()
