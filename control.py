import numpy as np
import matplotlib.pyplot as plt
from main import Params, dynamics, simulate


class ControlParams:
    # Targets
    z_target = 0.0
    zdot_target = -10.0
    x_target = 0.0
    xdot_target = 0.0
    
    # Outer loop gains (position/velocity)
    KP_x = 0.11
    KD_x = 0.66
    KP_z = 0.11
    KD_z = 0.66
    
    # Inner loop gains (attitude/torque)
    KP_tau = 1.7e7
    KD_tau = 5e6
    
    # Limits
    beta_max = np.deg2rad(20)
    TWR_min = 0.5
    TWR_max = 2.5


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
    e_w = -thetadot
    
    tau_des = cp.KP_tau * e_theta + cp.KD_tau * e_w
    
    # Gimbal angle from torque
    arg = tau_des / (p.iota * F_T)
    arg = np.clip(arg, -1.0, 1.0)
    delta = -np.arcsin(arg)
    
    return F_T, delta


def simulate_controlled(q0, t0, tf, dt, p, cp):
    n_steps = int(np.ceil((tf - t0) / dt)) + 1
    t_hist = np.zeros(n_steps)
    q_hist = np.zeros((n_steps, 6))
    F_hist = np.zeros(n_steps)
    delta_hist = np.zeros(n_steps)

    q = np.array(q0, dtype=float)
    t = t0
    delta_prev = 0.0

    for k in range(n_steps):
        t_hist[k] = t
        q_hist[k] = q
        
        F_T, delta = cascade_controller(t, q, p, cp, delta_prev)
        F_hist[k] = F_T
        delta_hist[k] = delta
        delta_prev = delta

        if k > 0:
            z, zdot = q[1], q[4]
            if z <= 0.0 and zdot <= 0.0:
                return t_hist[:k+1], q_hist[:k+1], F_hist[:k+1], delta_hist[:k+1]

        if t >= tf:
            return t_hist[:k+1], q_hist[:k+1], F_hist[:k+1], delta_hist[:k+1]

        q = q + dt * dynamics(t, q, F_T, delta, p)
        t += dt

    return t_hist, q_hist, F_hist, delta_hist


def run_landing():
    p = Params()
    cp = ControlParams()
    
    # Initial conditions: 300m up, drifting right, descending
    q0 = [50.0, 300.0, np.deg2rad(5), 10.0, -20.0, 0.0]
    
    t, qh, Fh, dh = simulate_controlled(q0, 0.0, 50.0, 0.01, p, cp)
    
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
