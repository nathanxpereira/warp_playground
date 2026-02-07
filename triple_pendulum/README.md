# Triple Inverted Pendulum

**Status**: 🚧 Coming Soon

## Overview

Balance three pendulums stacked on top of each other - a challenging multi-body control problem that extends the classic inverted pendulum to a more complex, higher-dimensional system.

## Planned Features

- **Advanced Control Strategies**:
  - Linear Quadratic Regulator (LQR) for linearized system
  - Model Predictive Control (MPC) for constraint handling
  - Reinforcement Learning (PPO, SAC) for learning-based control

- **Multi-Body Dynamics**: Accurate physics simulation of coupled pendulums

- **State Estimation**: Kalman filtering for noisy observations

- **Visualization**: Real-time rendering of the triple pendulum system

## System Description

**State Vector**:
```
[cart_position, cart_velocity,
 θ1, ω1,  # First pendulum (bottom)
 θ2, ω2,  # Second pendulum (middle)
 θ3, ω3]  # Third pendulum (top)
```

**Control**: Horizontal force applied to the cart

**Dimensions**: 8-dimensional state space, 1-dimensional control input

## Challenges

1. **High Dimensionality**: 8 states vs 4 for single pendulum
2. **Nonlinearity**: Stronger nonlinear coupling between pendulums
3. **Limited Region of Attraction**: Smaller basin of stability
4. **Actuation Limits**: Control saturation more critical
5. **Sensitivity**: Small disturbances can lead to failure

## Implementation Roadmap

- [ ] Derive equations of motion using Lagrangian mechanics
- [ ] Implement physics simulation in Isaac Sim
- [ ] Design LQR controller for swing-up and stabilization
- [ ] Add MPC for constraint handling
- [ ] Train RL agents for robust control
- [ ] Compare controller performance metrics
- [ ] Create visualization and analysis tools

## Getting Started

This project is currently under development. Check back soon or contribute to help build it!

## Contributing

Interested in implementing this project? We'd love your help! Areas that need work:

- Mathematical modeling and linearization
- Controller implementation
- Simulation environment setup
- Testing and benchmarking

See the [main README](../README.md) for contribution guidelines.

## References

- Åström, K. J., & Furuta, K. (2000). "Swinging up a pendulum by energy control"
- Graichen, K., & Zeitz, M. (2005). "Swing-up of the double pendulum on a cart by feedforward and feedback control"
- Control tutorials for MATLAB and Simulink - Triple Pendulum

---

**Back to [Main Repository](../README.md)**
