import torch
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
plt.switch_backend('tkagg')
plt.rc('font', family='serif', size=12)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'''
       \usepackage{amsmath,amsfonts}
       \renewcommand{\v}[1]{\boldsymbol{#1}}''')

def flatten(a):
    return np.array(a).flatten()

class InvertedPendulum:
    def __init__(
        self, dt=0.07, 
        tensor_args={
            'dtype': torch.float32, 
            'device': 'cpu'
            }
        ):
        self.tensor_args = tensor_args
        self.m = 1.0
        self.l = 1.0
        self.g = 9.8
       
    
        self.dt = dt

        # Plotting parameters
        self.radius = 0.1
        self.scale = 2.0

        self.fig = None
        self.ax = None
        self.rod = None
        self.bob = None
        self.title = None

        # Turn on interactive plotting
        plt.ion()

        # initialize x
        self.reset()

    def reset(self):
        self.x = torch.tensor([-torch.pi/2.0, 0.0], **self.tensor_args)
        return self.x

    def _calculate_next_state(self, theta, dtheta, u):
        """Calculate next state based on current angle, angular velocity, and input force."""
        dtheta_new = dtheta + (-3 * self.g / (2 * self.l) * torch.sin(theta + torch.pi) + 3. / (self.m * self.l**2) * u) * self.dt
        theta_new = theta + dtheta_new * self.dt
        return theta_new, dtheta_new

    def step(self, u=None):

        if u is None: # passive dynamics if no control action is given
            u = torch.tensor([0.0], **self.tensor_args)

        theta, dtheta = self.x.clone()
        theta_new, dtheta_new = self._calculate_next_state(theta, dtheta, u)
        self.x = torch.tensor([theta_new, dtheta_new], **self.tensor_args)
        return self.x

    def visualize(self, x=None, show_plot=True):
        if x is None:
            x = self.x.clone()
        theta, dtheta = x[0].item(), x[1].item()  # Convert tensor to scalar

        # Calculate pendulum position
        bx = np.array([0.0, self.l * math.sin(-theta)])
        by = np.array([0.0, self.l * math.cos(-theta)])
        
        if self.fig is None and self.ax is None:
            # Initialize the figure and axes
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_xlim(-self.l*1.2, self.l*1.2)
            self.ax.set_ylim(-self.l*1.2, self.l*1.2)
            self.ax.grid(True)

            # Draw anchor
            anchor = Circle((0,0), self.radius/self.scale, color='k')
            self.ax.add_patch(anchor)

            # Initialize rod line with a thicker line width
            self.rod, = self.ax.plot(bx, by, lw=4, color='k')  # Thick line for rod

            # Initialize bob with gradient coloring
            # Note: Gradient coloring on patches is complex; here we'll simulate it with a color map
            self.bob = Circle((bx[-1], by[-1]), self.radius, color='r')  # Bob as circle
            self.ax.add_patch(self.bob)

            # Title
            self.title = self.ax.set_title(f"$\\Theta$:{theta:.2f}   $\\dot{{\\Theta }}$:{dtheta:.2f}")
        else:
            # Update rod data
            self.rod.set_data(bx, by)
            # Update bob position
            self.bob.center = (bx[-1], by[-1])
            # Update title
            self.title.set_text(f"$\\Theta$:{theta:.2f}   $\\dot{{\\Theta }}$:{dtheta:.2f}")

        if show_plot:
            plt.draw()
            plt.pause(0.001)  # Pause to update the plot
        
# Testing the setup with default tensor arguments
tensor_args={
            'dtype': torch.float32, 
            'device': 'cpu'
            }

env = InvertedPendulum(tensor_args=tensor_args)
print("Initial State:", env.x)

# Run simulation steps with a dummy action
dummy_action = torch.tensor([0.0], **tensor_args)
for i in range(100):
    env.step(dummy_action)
    print(f"Time elapsed: {i * env.dt:.2f} seconds, State: {env.x}")

    # Visualize final state
    env.visualize()

# Keep the plot open after the loop finishes
plt.ioff()
plt.show()