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
    #TODO: Use Link and circle sizes that are the same order of maginitude as Franka
    links = [
        Link(length=1.5, fixed=True, angle_limits=None, fixedOrigin=True,
             circle_offsets=[0.4,1.1], circle_radii=[0.4,0.4], tensor_args=tensor_args),
        Link(length=1.0, fixed=False, angle_limits=(-math.pi, math.pi), fixedOrigin=False,
             circle_offsets=[0.3, 0.7], circle_radii=[0.35, 0.35], tensor_args=tensor_args),
        Link(length=0.7, fixed=False, angle_limits=(-math.pi/2, math.pi/2), fixedOrigin=False,
             circle_offsets=[0.2, 0.5], circle_radii=[0.3, 0.3], tensor_args=tensor_args)
    ]

#     start_angles = torch.tensor([np.pi/2.0, -0.6*np.pi, -math.pi/2], **tensor_args)
    start_angles = torch.tensor([np.pi/2.0, -1.0, -0.8], **tensor_args)
    env = DPlanarRobot(links=links, dt=0.05, tensor_args=tensor_args,starting_angle_config=start_angles, seed=0)

    # Reset environment with multiple particles for demonstration
    state = env.reset(num_particles=1)  # two parallel arms

    # plot robot state
    fig = env.plot_robot_state()
    fig.savefig("robot.pdf", bbox_inches='tight', pad_inches=0.1)

    # plot robot SDF (we would be  plotting the initial states since env.step hasn't been called yet)
    resolution  = 1000
    
    # sdf min and max makes it easier to see the SDF (simply applies np.clip(sdf_value, min, max) or similar)
    fig = env.plot_sdf(resolution=resolution, sdf_min=-0.5, sdf_max=0.5)
    fig.savefig("heatmap.pdf", bbox_inches='tight', pad_inches=0.1)
    
    print("Robot visualization saved as 'robot.pdf'")
    print("SDF heatmap saved as 'heatmap.pdf'")
    
    if interactive_mode:
        plt.show()
    else:
        print("Plots saved successfully! Open robot.pdf and heatmap.pdf to view results.")
