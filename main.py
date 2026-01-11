import numpy as np
import matplotlib.pyplot as plt


class Params:
    m = 5000.0
    g = 9.81
    iota = 3.0
    Iyy = 750000


def dynamics(t, q, F_T, delta, p):
    x, z, theta, xdot, zdot, thetadot = q
    beta = theta + delta
    xddot = (F_T / p.m) * np.sin(beta)
    zddot = (F_T / p.m) * np.cos(beta) - p.g
    thetaddot = -(F_T * p.iota / p.Iyy) * np.sin(delta)
    return np.array([xdot, zdot, thetadot, xddot, zddot, thetaddot])


def simulate(q0, t0, tf, dt, thrust_fn, delta_fn, p, stop_on_ground=True):
    n_steps = int(np.ceil((tf - t0) / dt)) + 1
    t_hist = np.zeros(n_steps)
    q_hist = np.zeros((n_steps, 6))

    q = np.array(q0, dtype=float)
    t = float(t0)

    for k in range(n_steps):
        t_hist[k] = t
        q_hist[k] = q

        if stop_on_ground and k > 0:
            z, zdot = q[1], q[4]
            if z <= 0.0 and zdot <= 0.0:
                return t_hist[:k+1], q_hist[:k+1]

        if t >= tf:
            return t_hist[:k+1], q_hist[:k+1]

        F_T = float(thrust_fn(t, q))
        delta = float(delta_fn(t, q))
        q = q + dt * dynamics(t, q, F_T, delta, p)
        t += dt

    return t_hist, q_hist


def test_free_fall():
    p = Params()
    q0 = [0.0, 100.0, 0.0, 5.0, 0.0, 0.0]
    
    t, qh = simulate(q0, 0.0, 3.0, 0.001, lambda t,q: 0.0, lambda t,q: 0.0, p, stop_on_ground=False)
    
    z_analytic = q0[1] + q0[4]*t - 0.5*p.g*t**2
    z_err = np.max(np.abs(qh[:,1] - z_analytic))
    
    print(f"Free fall: max z error = {z_err:.6f} m")
    
    plt.figure()
    plt.plot(t, qh[:,1], label='numeric')
    plt.plot(t, z_analytic, '--', label='analytic')
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.title('Free Fall')
    plt.legend()


def test_hover():
    p = Params()
    q0 = [0.0, 50.0, 0.0, 5.0, 0.0, 0.0]
    
    t, qh = simulate(q0, 0.0, 5.0, 0.01, lambda t,q: p.m*p.g, lambda t,q: 0.0, p, stop_on_ground=False)
    
    vx_drift = abs(qh[-1,3] - q0[3])
    print(f"Hover: vx drift = {vx_drift:.6f} m/s")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(t, qh[:,1])
    ax1.axhline(q0[1], color='r', linestyle='--', label='initial z')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('z [m]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    x_expected = q0[0] + q0[3] * t
    ax2.plot(t, qh[:,0], label='numeric')
    ax2.plot(t, x_expected, '--', color='r', label='expected')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('x [m]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()


def test_gimbal_torque():
    p = Params()
    q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    F_T = 2.0 * p.m * p.g
    delta = np.deg2rad(5.0)
    
    t, qh = simulate(q0, 0.0, 2.0, 0.001, lambda t,q: F_T, lambda t,q: delta, p, stop_on_ground=False)
    
    w_dot = -(F_T * p.iota / p.Iyy) * np.sin(delta)
    w_expected = w_dot * t
    theta_expected = 0.5 * w_dot * t**2
    
    print(f"Gimbal: expected w_dot = {w_dot:.4f} rad/s^2")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    ax1.plot(t, qh[:,5], label='numeric')
    ax1.plot(t, w_expected, '--', label='expected (linear)')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('w [rad/s]')
    ax1.set_title('Angular velocity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, qh[:,2], label='numeric')
    ax2.plot(t, theta_expected, '--', label='expected (parabolic)')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('theta [rad]')
    ax2.set_title('Pitch angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()


if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:]
    
    tests = {
        '--free-fall': test_free_fall,
        '--hover': test_hover,
        '--gimbal': test_gimbal_torque,
        '--landing': test_landing
    }
    
    selected = [k for k in tests.keys() if k in args]
    
    if '--verification' in args:
        for test_fn in tests.values():
            test_fn()
    elif selected:
        for test_name in selected:
            tests[test_name]()
    else:
        print("Usage: python main.py [OPTIONS]")
        print("\nOptions:")
        print("  --verification Run all tests")
        print("  --free-fall    Free fall validation")
        print("  --hover        Hover test")
        print("  --gimbal       Gimbal torque test")
        print("  --landing      Landing test")
    
    if selected or '--verification' in args:
        plt.show()
