"""
Visualize LQR tuning results and compare controller performance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(filepath="lqr_tuning_results.json"):
    """Load tuning results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_tuning_results(results):
    """Create comprehensive visualization of tuning results"""

    # Sort by total cost
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_cost'])

    # Extract data
    costs = [r['metrics']['total_cost'] for r in sorted_results]
    max_angles = [np.degrees(r['metrics']['max_pole_angle']) for r in sorted_results]
    final_angles = [np.degrees(r['metrics']['final_pole_angle']) for r in sorted_results]
    max_cart_pos = [r['metrics']['max_cart_pos'] for r in sorted_results]
    mean_forces = [r['metrics']['mean_abs_force'] for r in sorted_results]

    # Get Q and R values
    q_pole_angles = [np.diag(r['Q'])[2] for r in sorted_results]
    r_values = [r['R'][0][0] for r in sorted_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LQR Hyperparameter Tuning Results', fontsize=16)

    # 1. Total cost distribution
    ax = axes[0, 0]
    ax.bar(range(len(costs)), costs)
    ax.set_xlabel('Configuration (sorted by cost)')
    ax.set_ylabel('Total Cost')
    ax.set_title('Total Cost Distribution')
    ax.grid(True, alpha=0.3)

    # 2. Max pole angle
    ax = axes[0, 1]
    ax.bar(range(len(max_angles)), max_angles)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Max Pole Angle (degrees)')
    ax.set_title('Maximum Pole Deviation')
    ax.axhline(y=10, color='r', linestyle='--', label='10° threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final pole angle (steady-state error)
    ax = axes[0, 2]
    ax.bar(range(len(final_angles)), final_angles)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Final Pole Angle (degrees)')
    ax.set_title('Steady-State Error')
    ax.grid(True, alpha=0.3)

    # 4. Cost vs Q_pole_angle
    ax = axes[1, 0]
    scatter = ax.scatter(q_pole_angles, costs, c=r_values, cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Q[pole_angle]')
    ax.set_ylabel('Total Cost')
    ax.set_title('Cost vs Pole Angle Weight')
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax, label='R (control penalty)')
    ax.grid(True, alpha=0.3)

    # 5. Control effort
    ax = axes[1, 1]
    ax.bar(range(len(mean_forces)), mean_forces)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean Absolute Force (N)')
    ax.set_title('Average Control Effort')
    ax.grid(True, alpha=0.3)

    # 6. Max cart position
    ax = axes[1, 2]
    ax.bar(range(len(max_cart_pos)), max_cart_pos)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Max Cart Position (m)')
    ax.set_title('Maximum Cart Displacement')
    ax.axhline(y=1.0, color='r', linestyle='--', label='1m threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lqr_tuning_visualization.png', dpi=150)
    print("Visualization saved to lqr_tuning_visualization.png")
    plt.show()


def plot_pareto_frontier(results):
    """Plot Pareto frontier: cost vs control effort"""

    costs = [r['metrics']['total_cost'] for r in results]
    control_efforts = [r['metrics']['mean_abs_force'] for r in results]
    max_angles = [np.degrees(r['metrics']['max_pole_angle']) for r in results]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(control_efforts, costs, c=max_angles,
                        cmap='RdYlGn_r', s=100, alpha=0.6)
    ax.set_xlabel('Mean Control Effort (N)', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title('Pareto Frontier: Performance vs Control Effort', fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Max Pole Angle (degrees)', fontsize=12)

    # Annotate best configurations
    sorted_by_cost = sorted(results, key=lambda x: x['metrics']['total_cost'])
    for i, result in enumerate(sorted_by_cost[:3]):
        cost = result['metrics']['total_cost']
        effort = result['metrics']['mean_abs_force']
        ax.annotate(f'#{i+1}', (effort, cost), fontsize=10,
                   xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('lqr_pareto_frontier.png', dpi=150)
    print("Pareto frontier saved to lqr_pareto_frontier.png")
    plt.show()


def plot_parameter_heatmap(results):
    """Create heatmap showing how Q and R affect performance"""

    # Extract unique parameter values
    q_angles = sorted(list(set([np.diag(r['Q'])[2] for r in results])))
    r_vals = sorted(list(set([r['R'][0][0] for r in results])))

    # Create cost matrix
    cost_matrix = np.zeros((len(q_angles), len(r_vals)))
    angle_matrix = np.zeros((len(q_angles), len(r_vals)))

    for result in results:
        q_angle = np.diag(result['Q'])[2]
        r_val = result['R'][0][0]

        i = q_angles.index(q_angle)
        j = r_vals.index(r_val)

        cost_matrix[i, j] = result['metrics']['total_cost']
        angle_matrix[i, j] = np.degrees(result['metrics']['max_pole_angle'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Cost heatmap
    im1 = ax1.imshow(cost_matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks(range(len(r_vals)))
    ax1.set_yticks(range(len(q_angles)))
    ax1.set_xticklabels([f'{r:.3f}' for r in r_vals])
    ax1.set_yticklabels([f'{q:.1f}' for q in q_angles])
    ax1.set_xlabel('R (control penalty)', fontsize=12)
    ax1.set_ylabel('Q[pole_angle]', fontsize=12)
    ax1.set_title('Total Cost Heatmap', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Total Cost')

    # Angle heatmap
    im2 = ax2.imshow(angle_matrix, aspect='auto', cmap='RdYlGn_r')
    ax2.set_xticks(range(len(r_vals)))
    ax2.set_yticks(range(len(q_angles)))
    ax2.set_xticklabels([f'{r:.3f}' for r in r_vals])
    ax2.set_yticklabels([f'{q:.1f}' for q in q_angles])
    ax2.set_xlabel('R (control penalty)', fontsize=12)
    ax2.set_ylabel('Q[pole_angle]', fontsize=12)
    ax2.set_title('Max Pole Angle Heatmap', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Max Angle (degrees)')

    plt.tight_layout()
    plt.savefig('lqr_parameter_heatmap.png', dpi=150)
    print("Parameter heatmap saved to lqr_parameter_heatmap.png")
    plt.show()


if __name__ == "__main__":
    results_file = "lqr_tuning_results.json"

    if not Path(results_file).exists():
        print(f"Error: {results_file} not found!")
        print("Run 'python evaluate_lqr.py' first to generate tuning results.")
    else:
        results = load_results(results_file)
        print(f"Loaded {len(results)} configurations from {results_file}")

        print("\nGenerating visualizations...")
        plot_tuning_results(results)
        plot_pareto_frontier(results)
        plot_parameter_heatmap(results)

        print("\nAll visualizations complete!")
