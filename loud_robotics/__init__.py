"""
LOUD Robotics: A robotics package for Gaussian Process-based dynamics modeling.

This package provides tools for:
- Gaussian Process-based dynamics modeling
- Incremental Gaussian Processes
- Planar robot simulation and visualization
- Trajectory sampling and analysis
"""

__version__ = "0.1.0"
__author__ = "Sicelukwanda"
__email__ = "your.email@example.com"

# Import main components
from .environments import (
    InvertedPendulum,
    Link,
    DPlanarRobot,
    CircleObstacle,
    RectangleObstacle,
    trajectory_to_transitions,
    circle_sdf,
)

from .models import (
    IncrementalGP,
    GPList,
    IncrementalGPList,
)

__all__ = [
    # Environment components
    "InvertedPendulum",
    "Link", 
    "DPlanarRobot",
    "CircleObstacle",
    "RectangleObstacle",
    "trajectory_to_transitions",
    "circle_sdf",
    # Model components
    "IncrementalGP",
    "GPList",
    "IncrementalGPList",
]
