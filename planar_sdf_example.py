from environments import Link, DPlanarRobot
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# plotting options
plt.switch_backend("tkagg")
plt.rc("font", family="serif", size=14)
plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"""
       \usepackage{amsmath,amsfonts}
       \renewcommand{\v}[1]{\boldsymbol{#1}}""",
)

# Example usage:
# Define a 3-link arm: first link fixedOrigin and actuated, second link fixed, third link actuated
# Just as a demonstration:
if __name__ == "__main__":
    tensor_args = {"dtype": torch.float32, "device": "cpu"}

    # Create a 3-link planar arm
    # Let's say each link is 1.0 length, 
    # The first link is actuated and has 2 circles at offsets [0.3, 0.7] with radii [0.05, 0.05]
    # The second link is fixed, no circles
    # The third link is actuated and has 3 circles at offsets [0.2, 0.5, 0.8] with radii [0.05, 0.05, 0.05]
    #TODO: Use Link and circle sizes that are the same order of maginitude as Franka
    links = [
        Link(length=1.0, fixed=False, angle_limits=(-math.pi, math.pi), fixedOrigin=True,
             circle_offsets=[0.3, 0.7], circle_radii=[0.5, 0.4], tensor_args=tensor_args),
        Link(length=1.0, fixed=True, angle_limits=None, fixedOrigin=False,
             circle_offsets=[0.5], circle_radii=[0.4], tensor_args=tensor_args),
        Link(length=1.0, fixed=False, angle_limits=(-math.pi/2, math.pi/2), fixedOrigin=False,
             circle_offsets=[0.2, 0.5, 0.8], circle_radii=[0.5, 0.5, 0.5], tensor_args=tensor_args)
    ]

    env = DPlanarRobot(links=links, dt=0.05, tensor_args=tensor_args, seed=0)

    # Reset environment with multiple particles for demonstration
    state = env.reset(num_particles=2)  # two parallel arms

    # plot robot SDF (we would be  plotting the initial states since env.step hasn't been called yet)
    resolution  = 200
    
    # sdf min and max makes it easier to see the SDF (simply applies np.clip(sdf_value, min, max) or similar)
    env.plot_sdf(resolution=resolution, sdf_min=-0.5, sdf_max=0.5)

    # Keep the plot open at the end
    plt.ioff()
    plt.show()
