# Warp Playground

A collection of robotics simulation projects using NVIDIA Isaac Sim and Warp for physics-based control, reinforcement learning, and robotics research.

## 🎯 Projects

### 1. [Cartpole (Inverted Pendulum)](cartpole/)
**Status**: ✅ Active

Classic control theory problem: balance an inverted pendulum on a moving cart.

- **Controllers**: LQR (optimal control), PID
- **Features**: Parallel environments, hyperparameter tuning, performance visualization
- **Difficulty**: Beginner

[→ View Project](cartpole/)

---

### 2. Triple Inverted Pendulum
**Status**: 🚧 Coming Soon

Balance three pendulums stacked on top of each other - a challenging multi-body control problem.

- **Planned Controllers**: LQR, MPC, RL
- **Focus**: Multi-body dynamics, advanced control strategies
- **Difficulty**: Advanced

---

### 3. Humanoid Locomotion
**Status**: 🚧 Coming Soon

Train a humanoid robot to walk using reinforcement learning.

- **Planned Approaches**: PPO, SAC, model-based RL
- **Focus**: High-dimensional control, sim-to-real transfer
- **Difficulty**: Expert

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **NVIDIA GPU**: Required for Isaac Sim and CUDA-accelerated PyTorch
- **CUDA**: 12.8 or compatible version
- **conda** or **venv**: For environment management
- **Disk Space**: ~15GB for Isaac Sim and dependencies

### Installation

#### Option 1: Full Installation (GPU + CUDA)

1. **Clone the repository**:
```bash
git clone https://github.com/nathanxpereira/warp_playground/warp-playground.git
cd warp-playground
```

2. **Create and activate a conda environment**:
```bash
conda create -p ./.venv python=3.11
conda activate ./.venv
```

3. **Install all dependencies** (PyTorch with CUDA + Isaac Sim):
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

This installs:
- PyTorch 2.10.0 with CUDA 13.0 support (~2GB)
- NVIDIA Isaac Sim 5.1.0 (~10GB)
- Additional dependencies (numpy, matplotlib, scipy)

4. **Generate VSCode settings**:
```bash
python -m isaacsim --generate-vscode-settings
```

### Verify Installation

```bash
# Test PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Isaac Sim
python -c "from isaacsim import SimulationApp; print('Isaac Sim installed successfully!')"
```

Expected output:
```
PyTorch: 2.10.0+cu130
CUDA available: True
Isaac Sim installed successfully!
```

### Run Your First Project

```bash
# Navigate to the cartpole project
cd cartpole

# Run the simulation
python cartpole.py
```

## 📁 Repository Structure

```
warp_playground/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── cartpole/                   # Inverted pendulum project
│   ├── README.md
│   ├── cartpole.py
│   ├── config.py
│   ├── controllers.py
├── triple_pendulum/            # Triple pendulum (coming soon)
└── humanoid_walk/              # Humanoid locomotion (coming soon)
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code with black
black .

# Lint with ruff
ruff check .

# Type checking
mypy .
```

### CI/CD

GitHub Actions automatically runs tests, linting, and type checking on all pull requests. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## 📚 Learning Resources

### Control Theory
- [Introduction to Linear Quadratic Regulator (LQR)](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator)
- [PID Controller Tuning](https://en.wikipedia.org/wiki/PID_controller)
- [Model Predictive Control](https://en.wikipedia.org/wiki/Model_predictive_control)

### Reinforcement Learning
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

### NVIDIA Isaac Sim
- [Official Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [Python API Reference](https://docs.omniverse.nvidia.com/py/isaacsim/)
- [Tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorials.html)


## 🙏 Acknowledgments

- NVIDIA for Isaac Sim and Warp
- The control theory and robotics research community
- All contributors to this project

## 📧 Contact

Nathan Pereira - [GitHub Profile](https://github.com/nathanxpereira)

Project Link: [https://github.com/nathanxpereira/warp-playground](https://github.com/yourusername/warp-playground)

