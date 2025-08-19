from loud_robotics import Link, DPlanarRobot
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# plotting options
try:
    import tkinter
    plt.switch_backend("tkagg")
    interactive_mode = True
except:
    try:
        plt.switch_backend("Qt5Agg")
        interactive_mode = True
    except:
        plt.switch_backend("Agg")
        interactive_mode = False
        print("Using non-interactive backend - plots will be saved as files")

plt.rc("font", family="serif", size=14)
# Try to use LaTeX if available, otherwise fall back to regular text
try:
    plt.rc("text", usetex=True)
    plt.rc(
        "text.latex",
        preamble=r"""
           \usepackage{amsmath,amsfonts}
           \renewcommand{\v}[1]{\boldsymbol{#1}}""",
    )
except Exception:
    print("LaTeX not available, using regular text rendering")
    plt.rc("text", usetex=False)
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
    steps = 50

    for t in range(steps):
        # Run simulation with zero input torques
        u = torch.zeros((env.num_particles, env.action_dim), **tensor_args)
        env.step(u)
        env.visualize(show_plot=True, show_circles=True)

    # Example of computing gradients w.r.t. joint angles:
    # Suppose we define a simple cost function: sum of squared end-effector positions (just as a demo).
    # We'll pick the first particle and the last link (end-effector).
    angles = env.x[:, :len(env.actuated_indices)]  # current actuated angles
    full_angles = env._build_full_angle_vector(angles)
    origins, endpoints, circle_positions = env.forward_kinematics(full_angles)

    # Let's say we want to minimize the distance of the end-effector of particle 0 from the origin.
    # cost = (ex^2 + ey^2) for the end-effector of particle 0
    ex, ey = endpoints[0, -1, 0], endpoints[0, -1, 1]  # end-effector of last link of the first particle
    cost = ex**2 + ey**2

    # Enable gradient computation
    # angles currently might not have requires_grad, so let's re-compute cost in a graph with gradient.
    angles_detached = angles.detach().clone().requires_grad_(True)
    full_angles_detached = env._build_full_angle_vector(angles_detached)
    origins_det, endpoints_det, _ = env.forward_kinematics(full_angles_detached)
    ex_det, ey_det = endpoints_det[0, -1, 0], endpoints_det[0, -1, 1]
    cost_det = ex_det**2 + ey_det**2

    cost_det.backward()  # compute gradients w.r.t. angles_detached
    print("Gradients of cost w.r.t. actuated angles:", angles_detached.grad)

    # Keep the plot open at the end
    plt.ioff()
    if interactive_mode:
        plt.show()
    else:
        print("Gradient example completed! Use interactive mode to see visualization.")
