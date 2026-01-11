import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from control import ControlParams, simulate_controlled, run_landing
from main import Params

def run_comparison():
    # Run Python simulation
    p = Params()
    cp = ControlParams()
    q0 = [20.0, 300.0, np.deg2rad(2), 5.0, -10.0, 0.0]
    
    t_py, q_py, F_py, delta_py, _, _ = simulate_controlled(q0, 0, 50, 0.001, p, cp)
    
    # Find landing index
    land_idx = np.where(q_py[:, 1] <= 0)[0]
    if len(land_idx) > 0:
        land_idx = land_idx[0]
    else:
        land_idx = len(t_py) - 1
    
    t_py = t_py[:land_idx+1]
    q_py = q_py[:land_idx+1]
    delta_py = delta_py[:land_idx+1]
    F_py = F_py[:land_idx+1]
    TWR_py = F_py / (p.m * p.g)
    
    # Load C++ results
    try:
        cpp_data = pd.read_csv('cpp/cpp_trajectory.csv')
        has_cpp = True
    except:
        print("C++ data not found. Run export_trajectory.exe first.")
        has_cpp = False
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Python vs C++ Implementation Comparison', fontsize=14, fontweight='bold')
    
    # Colors
    py_color = '#2196F3'  # Blue
    cpp_color = '#FF5722'  # Orange
    
    # 1. Trajectory (x vs z)
    ax = axes[0, 0]
    ax.plot(q_py[:, 0], q_py[:, 1], color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['x'], cpp_data['z'], '--', color=cpp_color, linewidth=2, label='C++')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 25)
    
    # 2. Altitude vs time
    ax = axes[0, 1]
    ax.plot(t_py, q_py[:, 1], color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['t'], cpp_data['z'], '--', color=cpp_color, linewidth=2, label='C++')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('z [m]')
    ax.set_title('Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Horizontal position vs time
    ax = axes[1, 0]
    ax.plot(t_py, q_py[:, 0], color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['t'], cpp_data['x'], '--', color=cpp_color, linewidth=2, label='C++')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('x [m]')
    ax.set_title('Horizontal Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Pitch angle vs time
    ax = axes[1, 1]
    ax.plot(t_py, np.rad2deg(q_py[:, 2]), color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['t'], np.rad2deg(cpp_data['theta']), '--', color=cpp_color, linewidth=2, label='C++')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('θ [deg]')
    ax.set_title('Pitch Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Gimbal angle vs time
    ax = axes[2, 0]
    ax.plot(t_py, np.rad2deg(delta_py), color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['t'], np.rad2deg(cpp_data['delta']), '--', color=cpp_color, linewidth=2, label='C++')
    ax.axhline(y=10, color='r', linestyle=':', alpha=0.5, label='Limit')
    ax.axhline(y=-10, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('δ [deg]')
    ax.set_title('Gimbal Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. TWR vs time
    ax = axes[2, 1]
    ax.plot(t_py, TWR_py, color=py_color, linewidth=2, label='Python')
    if has_cpp:
        ax.plot(cpp_data['t'], cpp_data['TWR'], '--', color=cpp_color, linewidth=2, label='C++')
    ax.axhline(y=1.3, color='r', linestyle=':', alpha=0.5, label='Limit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('TWR')
    ax.set_title('Thrust-to-Weight Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('python_cpp_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: python_cpp_comparison.png")
    
    # Print numerical comparison
    print("\n" + "="*60)
    print("NUMERICAL COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'Python':>15} {'C++':>15} {'Diff':>12}")
    print("-"*60)
    
    py_land_t = t_py[-1]
    py_final_x = q_py[-1, 0]
    py_final_vz = q_py[-1, 4]
    py_final_vx = q_py[-1, 3]
    
    if has_cpp:
        cpp_land_t = cpp_data['t'].iloc[-1]
        cpp_final_x = cpp_data['x'].iloc[-1]
        cpp_final_vz = cpp_data['zdot'].iloc[-1]
        cpp_final_vx = cpp_data['xdot'].iloc[-1]
        
        print(f"{'Landing time [s]':<25} {py_land_t:>15.4f} {cpp_land_t:>15.4f} {abs(py_land_t-cpp_land_t):>12.6f}")
        print(f"{'Final x [m]':<25} {py_final_x:>15.4f} {cpp_final_x:>15.4f} {abs(py_final_x-cpp_final_x):>12.6f}")
        print(f"{'Final vz [m/s]':<25} {py_final_vz:>15.4f} {cpp_final_vz:>15.4f} {abs(py_final_vz-cpp_final_vz):>12.6f}")
        print(f"{'Final vx [m/s]':<25} {py_final_vx:>15.4f} {cpp_final_vx:>15.4f} {abs(py_final_vx-cpp_final_vx):>12.6f}")
        
        # Compute max difference across trajectory
        min_len = min(len(t_py), len(cpp_data))
        x_diff = np.max(np.abs(q_py[:min_len, 0] - cpp_data['x'].values[:min_len]))
        z_diff = np.max(np.abs(q_py[:min_len, 1] - cpp_data['z'].values[:min_len]))
        theta_diff = np.max(np.abs(q_py[:min_len, 2] - cpp_data['theta'].values[:min_len]))
        
        print("-"*60)
        print(f"{'Max |Δx| [m]':<25} {x_diff:>42.6f}")
        print(f"{'Max |Δz| [m]':<25} {z_diff:>42.6f}")
        print(f"{'Max |Δθ| [rad]':<25} {theta_diff:>42.6f}")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    run_comparison()
