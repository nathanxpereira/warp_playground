"""
LQR Performance Evaluation and Hyperparameter Tuning

This script helps you:
1. Measure the performance of your LQR controller over N timesteps
2. Systematically tune Q and R matrices to optimize performance
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import torch
from isaacsim.core.api import World
from cartpole import CartPole
from controller import LQRController
import matplotlib.pyplot as plt
from itertools import product
import json

class PerformanceMetrics:
    """Calculate various performance metrics for the cartpole controller"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cart_positions = []
        self.pole_angles = []
        self.forces = []
        self.timesteps = 0

    def update(self, cart_pos, pole_angle, force):
        """Update metrics with current state"""
        self.cart_positions.append(cart_pos)
        self.pole_angles.append(pole_angle)
        self.forces.append(force)
        self.timesteps += 1

    def compute_metrics(self):
        """Compute aggregate performance metrics"""
        cart_positions = np.array(self.cart_positions)
        pole_angles = np.array(self.pole_angles)
        forces = np.array(self.forces)

        metrics = {
            # Primary objectives
            "mean_abs_pole_angle": np.mean(np.abs(pole_angles)),
            "max_pole_angle": np.max(np.abs(pole_angles)),
            "final_pole_angle": np.abs(pole_angles[-1]),
            "pole_angle_std": np.std(pole_angles),

            "mean_abs_cart_pos": np.mean(np.abs(cart_positions)),
            "max_cart_pos": np.max(np.abs(cart_positions)),
            "final_cart_pos": np.abs(cart_positions[-1]),

            # Control effort
            "mean_abs_force": np.mean(np.abs(forces)),
            "max_force": np.max(np.abs(forces)),
            "total_force": np.sum(np.abs(forces)),

            # Settling metrics
            "settling_time_pole": self._settling_time(pole_angles, threshold=0.01),
            "settling_time_cart": self._settling_time(cart_positions, threshold=0.05),

            # Overall cost (weighted sum)
            "total_cost": self._compute_total_cost(cart_positions, pole_angles, forces)
        }

        return metrics

    def _settling_time(self, values, threshold=0.01):
        """Find timestep when values settle below threshold"""
        for i in range(len(values) - 1, -1, -1):
            if np.abs(values[i]) > threshold:
                return i + 1
        return 0

    def _compute_total_cost(self, cart_pos, pole_angles, forces):
        """Compute LQR-like cost function"""
        # This reflects what LQR is trying to minimize
        state_cost = np.sum(cart_pos**2 + 5*pole_angles**2)
        control_cost = np.sum(forces**2)
        return state_cost + control_cost


def run_simulation(controller, num_steps=1000, render=False, initial_angle=5.0):
    """Run simulation and collect performance data"""

    my_world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene",
                     backend="torch", device="cuda:0")
    my_task = CartPole(name="cartpole", num_envs=1)
    my_task.start_angle = initial_angle
    my_world.add_task(my_task)
    my_world.reset()

    metrics = PerformanceMetrics()

    # Initialize simulation
    my_world.reset(soft=True)
    my_task.reset_state()
    for _ in range(3):
        my_world.step(render=False)
    controller.reset()

    # Run simulation
    for step in range(num_steps):
        observations = my_world.get_observations()

        # Extract state for metrics
        cart_pos = observations["cart_position"][0, 0].cpu().item()
        pole_rot = observations["pole_rotation"][0]
        pole_angle = 2 * torch.atan2(pole_rot[1], pole_rot[3]).cpu().item()

        # Compute control
        force = controller.compute_control(observations, my_world.device)
        force_value = force[0].cpu().item() if isinstance(force, torch.Tensor) else force

        # Update metrics
        metrics.update(cart_pos, pole_angle, force_value)

        # Apply control and step
        my_task.apply_control(force)
        my_world.step(render=render)

    my_world.stop()
    return metrics


def tune_lqr_hyperparameters(param_grid, num_steps=1000, initial_angle=5.0):
    """
    Tune LQR Q and R matrices using grid search

    param_grid: dict with keys like 'q_cart_pos', 'q_cart_vel', 'q_pole_angle', 'q_pole_vel', 'r'
    """

    results = []

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    total = len(combinations)
    print(f"Testing {total} parameter combinations...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Create Q and R matrices
        Q = np.diag([
            params.get('q_cart_pos', 1.0),
            params.get('q_cart_vel', 1.0),
            params.get('q_pole_angle', 1.0),
            params.get('q_pole_vel', 1.0)
        ])
        R = np.array([[params.get('r', 1.0)]])

        # Create controller with custom Q, R
        controller = LQRController(m_cart=1.0, m_pole=0.1, l=1.0)
        # Override the default Q, R and recompute gains
        from scipy.linalg import solve_continuous_are
        A = np.array([[0, 1, 0, 0],
                      [0, 0, -6*0.1*9.81/(4*1.0+0.1), 0],
                      [0, 0, 0, 1],
                      [0, 0, (2*9.81*1.1)/(1.0*(4*1.0+0.1)), 0]])
        B = np.array([[0], [4/(4*1.0+0.1)], [0], [-6/(1.0*(4*1.0+0.1))]])
        P = solve_continuous_are(A, B, Q, R)
        controller.K = (np.linalg.inv(R) @ B.T @ P).flatten()

        # Run simulation
        try:
            metrics_obj = run_simulation(controller, num_steps=num_steps,
                                        render=False, initial_angle=initial_angle)
            metrics = metrics_obj.compute_metrics()

            result = {
                'params': params,
                'Q': Q.tolist(),
                'R': R.tolist(),
                'K': controller.K.tolist(),
                'metrics': metrics
            }
            results.append(result)

            print(f"[{i+1}/{total}] Q_diag={[params.get(k, 1.0) for k in ['q_cart_pos', 'q_cart_vel', 'q_pole_angle', 'q_pole_vel']]}, "
                  f"R={params.get('r', 1.0):.2f} -> Cost={metrics['total_cost']:.2f}, "
                  f"Max angle={np.degrees(metrics['max_pole_angle']):.2f}°")

        except Exception as e:
            print(f"[{i+1}/{total}] FAILED: {e}")

    return results


def analyze_results(results, save_path="lqr_tuning_results.json"):
    """Analyze and visualize tuning results"""

    # Save raw results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")

    # Sort by total cost
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_cost'])

    print("\n" + "="*80)
    print("TOP 5 PARAMETER CONFIGURATIONS")
    print("="*80)

    for i, result in enumerate(sorted_results[:5]):
        print(f"\n#{i+1} Configuration:")
        print(f"  Q diagonal: {np.diag(result['Q'])}")
        print(f"  R: {result['R'][0][0]:.3f}")
        print(f"  K gains: {result['K']}")
        print(f"  Total cost: {result['metrics']['total_cost']:.2f}")
        print(f"  Max pole angle: {np.degrees(result['metrics']['max_pole_angle']):.2f}°")
        print(f"  Final pole angle: {np.degrees(result['metrics']['final_pole_angle']):.4f}°")
        print(f"  Max cart position: {result['metrics']['max_cart_pos']:.3f} m")
        print(f"  Mean control force: {result['metrics']['mean_abs_force']:.2f} N")

    return sorted_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Single evaluation mode
        print("Evaluating current LQR controller...")
        controller = LQRController(m_cart=1.0, m_pole=0.1, l=1.0)
        metrics_obj = run_simulation(controller, num_steps=1000, render=False, initial_angle=5.0)
        metrics = metrics_obj.compute_metrics()

        print("\nPerformance Metrics:")
        print("="*50)
        for key, value in metrics.items():
            if 'angle' in key:
                print(f"{key:30s}: {np.degrees(value):10.4f}°")
            else:
                print(f"{key:30s}: {value:10.4f}")

    else:
        # Hyperparameter tuning mode
        print("Starting LQR hyperparameter tuning...")

        # Define search grid
        # These weights determine what the controller prioritizes
        param_grid = {
            'q_cart_pos':   [0.1, 1.0, 10.0],      # Weight on cart position
            'q_cart_vel':   [0.1, 1.0, 5.0],       # Weight on cart velocity
            'q_pole_angle': [1.0, 10.0, 50.0],     # Weight on pole angle (most important!)
            'q_pole_vel':   [0.1, 1.0, 5.0],       # Weight on pole angular velocity
            'r':            [0.01, 0.1, 1.0]       # Weight on control effort (penalty for large forces)
        }

        results = tune_lqr_hyperparameters(param_grid, num_steps=500, initial_angle=5.0)
        best_configs = analyze_results(results)

        print("\n" + "="*80)
        print("Tuning complete! Check lqr_tuning_results.json for full results.")
        print("="*80)

    simulation_app.close()
