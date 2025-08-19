# LOUD Robotics

A Python package for Gaussian Process-based dynamics modeling and planar robot simulation.

## Features

- **Gaussian Process Models**: Standard and Incremental Gaussian Processes for dynamics learning
- **Planar Robot Simulation**: Configurable multi-link planar robot simulation with collision detection
- **Trajectory Sampling**: Tools for trajectory generation and analysis
- **Visualization**: Built-in plotting and visualization utilities

## Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/Sicelukwanda/loud-robotics.git
cd loud-robotics

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/Sicelukwanda/loud-robotics.git
cd loud-robotics

# Install dependencies
pip install -e .
```

## Quick Start

### Basic Gaussian Process Usage

```python
import numpy as np
import GPy
from loud_robotics import GPList

# Generate synthetic data
X = np.linspace(0, 10, 50).reshape(-1, 1)
Y1 = np.sin(X) + 0.1 * np.random.randn(50, 1)
Y2 = np.cos(X) + 0.1 * np.random.randn(50, 1)
Y = np.hstack((Y1, Y2))

# Create GP models
kernel_list = [GPy.kern.RBF(input_dim=1) for _ in range(Y.shape[1])]
gp_list = GPList(X, Y, kernel_list, noise_var=1e-6)

# Optimize and predict
gp_list.optimize({'optimizer': 'bfgs', 'max_iters': 1000})
X_new = np.linspace(0, 10, 100).reshape(-1, 1)
mu, var = gp_list.predict(X_new)
```

### Planar Robot Simulation

```python
import torch
import math
from loud_robotics import Link, DPlanarRobot

# Define robot links
links = [
    Link(length=1.0, fixed=False, angle_limits=(-math.pi, math.pi), 
         fixedOrigin=True, circle_offsets=[0.3, 0.7], circle_radii=[0.1, 0.1]),
    Link(length=1.0, fixed=False, angle_limits=(-math.pi/2, math.pi/2), 
         fixedOrigin=False, circle_offsets=[0.5], circle_radii=[0.1])
]

# Create robot environment
env = DPlanarRobot(links=links, dt=0.05)

# Reset and simulate
state = env.reset(num_particles=1)
for _ in range(100):
    action = torch.zeros((env.num_particles, env.action_dim))
    env.step(action)
    env.visualize(show_plot=True)
```

## Package Structure

```
loud_robotics/
├── environments/          # Robot simulation environments
│   ├── dynamical_system.py   # Base dynamical system classes
│   ├── planar_robot.py       # Planar robot simulation
│   └── dynamics_utils.py     # Utility functions
├── models/                # Gaussian Process models
│   ├── IGP.py               # Incremental Gaussian Process
│   ├── list_models.py       # Multi-output GP classes
│   └── utils.py             # Model utilities
└── examples/              # Example scripts
    ├── trajectory_sampling.py
    ├── planar_grad_example.py
    └── ...
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `trajectory_sampling.py`: Demonstrates trajectory sampling with GP models
- `planar_grad_example.py`: Shows gradient computation for planar robots
- `planar_sdf_example.py`: Visualizes signed distance fields
- `gp_list_example.py`: Basic GP model usage
- `igp_example.py`: Incremental GP demonstration

To run an example:

```bash
# Activate the environment
uv run python examples/gp_list_example.py
```

## Development

### Setting up development environment

```bash
# Clone and install in development mode
git clone https://github.com/Sicelukwanda/loud-robotics.git
cd loud-robotics
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black loud_robotics/
uv run isort loud_robotics/
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=loud_robotics
```

## Dependencies

- **torch**: PyTorch for tensor operations and automatic differentiation
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **GPy**: Gaussian Process library
- **scipy**: Scientific computing utilities

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{loud_robotics,
  author = {Sicelukwanda},
  title = {LOUD Robotics: Gaussian Process-based Robotics Package},
  url = {https://github.com/Sicelukwanda/loud-robotics},
  year = {2025}
}
```
