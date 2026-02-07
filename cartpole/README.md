# Cartpole (Inverted Pendulum) Project

A cart-pole (inverted pendulum) simulation using NVIDIA Isaac Sim's physics engine with multiple control strategies.

## Overview

This project implements a classic control theory problem: balancing an inverted pendulum on a cart. The cart can move horizontally, and the goal is to keep the pole upright by applying forces to the cart.

## Features

- **Physics-based simulation** using NVIDIA Isaac Sim
- **Multiple control algorithms**:
  - Linear Quadratic Regulator (LQR) - optimal control
  - Proportional-Integral-Derivative (PID) control
- **Parallel environments** for efficient testing
- **Hyperparameter tuning** framework for controller optimization
- **Performance visualization** tools
- **Mathematical documentation** of system dynamics

## System Description

**State Vector**: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`

**Control**: Horizontal force applied to the cart

See [cartpole_system.md](cartpole_system.md) for the complete mathematical derivation using Lagrangian mechanics.

## Project Structure

```
cartpole/
├── README.md                       # This file
├── cartpole.py                     # Main simulation entry point
├── config.py                       # Physical and simulation configuration
├── cartpole_system.md              # Mathematical derivation
├── controllers/                    # Controller implementations
│   ├── __init__.py
│   ├── abstract_controller.py     # Base class for all controllers
│   ├── LQR.py                      # Linear Quadratic Regulator
│   └── PID.py                      # PID controller
├── analysis/                       # Performance evaluation tools
│   ├── evaluate_lqr.py            # Hyperparameter tuning
│   └── visualize_results.py       # Result visualization
├── cartpole.usd                    # USD scene file (generated)
└── inverted_pendulum_highlevel.usd # Alternative visualization scene
```

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA Isaac Sim 5.0.0
- PyTorch (with CUDA support)

See the [main README](../README.md) for installation instructions.

### Running the Simulation

**Basic simulation with LQR controller**:
```bash
python cartpole.py
```

**Run with visualization**:
Edit [config.py](config.py) and set `render=True` in `SimulationConfig`.

### Tuning Controller Parameters

**LQR hyperparameter tuning**:
```bash
python analysis/evaluate_lqr.py
```

This performs a grid search over Q and R matrix parameters and saves results to `lqr_tuning_results.json`.

**Evaluate a single controller**:
```bash
python analysis/evaluate_lqr.py evaluate
```

**Visualize tuning results**:
```bash
python analysis/visualize_results.py
```

Generates plots showing:
- Performance metrics over time
- Pareto frontier (cost vs control effort)
- Parameter heatmaps

## Configuration

Edit [config.py](config.py) to modify:

### Physical Parameters (`CartPolePhysicsConfig`)
- `cart_mass`: Mass of the cart (kg)
- `pole_mass`: Mass of the pole (kg)
- `pole_length`: Length of the pole (m)
- `gravity`: Gravitational acceleration (m/s²)

### Simulation Parameters (`SimulationConfig`)
- `warmup_steps`: Number of initial steps before control
- `render`: Enable/disable visualization
- `physics_dt`: Physics timestep (seconds)

## Controllers

### LQR (Linear Quadratic Regulator)

Optimal controller that minimizes a quadratic cost function:
```
J = ∫ (x^T Q x + u^T R u) dt
```

**Parameters**:
- `Q`: State cost matrix (4x4) - penalizes deviations from equilibrium
- `R`: Control cost matrix (1x1) - penalizes control effort

**Tuning guidelines**:
- Increase Q diagonal elements to tighten control of specific states
- Increase R to reduce control effort (smoother but potentially slower response)

### PID Controller

Classic feedback controller with three terms:
- **P** (Proportional): Responds to current error
- **I** (Integral): Responds to accumulated error
- **D** (Derivative): Responds to rate of change

**Parameters**:
- `kp`: Proportional gain
- `ki`: Integral gain
- `kd`: Derivative gain

## Performance Metrics

The evaluation framework tracks:
- **Stability**: Mean and max pole angle from vertical
- **Control effort**: Mean and max applied forces
- **Settling time**: Time to reach steady state
- **LQR cost**: Total quadratic cost (for LQR controller)

## Adding New Controllers

1. Create a new file in `controllers/`
2. Inherit from `Controller` in [abstract_controller.py](controllers/abstract_controller.py)
3. Implement `reset()` and `compute_control()` methods
4. Update [cartpole.py](cartpole.py) to use your controller

Example:
```python
from controllers.abstract_controller import Controller
import torch

class MyController(Controller):
    def reset(self):
        # Initialize any internal state
        pass

    def compute_control(self, observations, device="cuda"):
        # Extract state from observations
        # Compute and return control force as torch.Tensor
        return control_force
```

## References

- [Inverted Pendulum - Wikipedia](https://en.wikipedia.org/wiki/Inverted_pendulum)
- [LQR - Wikipedia](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator)
- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)

## Future Work

- [ ] Add Model Predictive Control (MPC)
- [ ] Implement reinforcement learning controllers
- [ ] Add cart position constraints
- [ ] Support double and triple inverted pendulums
- [ ] Add disturbance rejection tests
