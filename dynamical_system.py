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

        # plotting
        self.radius = 0.1
        self.scale = 1.5
    
        self.dt = dt

        # Initialize plot attributes
        self.fig = None
        self.ax_pendulum = None
        self.ax_theta = None
        self.ax_dtheta = None
        self.rod = None
        self.bob = None
        self.title = None

        # Data for plots
        self.time_data = []
        self.theta_data = []
        self.dtheta_data = []

        # Turn on interactive plotting
        plt.ion()
    
        # initializer
        self.reset()

    def reset(self):
        self.x = torch.tensor([-torch.pi/2.0, 0.0], **self.tensor_args)
        self.time_elapsed = 0.0
        self.time_data = [self.time_elapsed]
        self.theta_data = [self.x[0].item()]
        self.dtheta_data = [self.x[1].item()]
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
        self.time_elapsed += self.dt

        # Update data lists
        self.time_data.append(self.time_elapsed)
        self.theta_data.append(theta_new.item())
        self.dtheta_data.append(dtheta_new.item())

        return self.x

    def visualize(self, x=None, show_plot=True):
        if x is None:
            x = self.x.clone()
        theta, dtheta = x[0].item(), x[1].item()  # Convert tensor to scalar

        # Calculate pendulum position
        bx = np.array([0.0, self.l * math.sin(-theta)])
        by = np.array([0.0, self.l * math.cos(-theta)])
        
        if self.fig is None:
            # Initialize the figure and axes
            self.fig = plt.figure(figsize=(10, 6))
            gs = self.fig.add_gridspec(2, 2)
            
            # Pendulum plot
            self.ax_pendulum = self.fig.add_subplot(gs[:, 0])
            self.ax_pendulum.set_aspect('equal')
            self.ax_pendulum.set_xlim(-self.l*1.2, self.l*1.2)
            self.ax_pendulum.set_ylim(-self.l*1.2, self.l*1.2)
            self.ax_pendulum.grid(True)

            # Draw anchor
            anchor = Circle((0,0), self.radius/self.scale, color='k')
            self.ax_pendulum.add_patch(anchor)

            # Initialize rod line with a thicker line width
            self.rod, = self.ax_pendulum.plot(bx, by, lw=4, color='k')  # Thick line for rod

            # Initialize bob
            self.bob = Circle((bx[-1], by[-1]), self.radius, color='r')  # Bob as circle
            self.ax_pendulum.add_patch(self.bob)

            # Title
            self.title = self.ax_pendulum.set_title(f"Time: {self.time_elapsed:.2f}s  $\\Theta$:{theta:.2f} rad   $\\dot{{\\Theta }}$:{dtheta:.2f} rad/s")

            # Theta plot
            self.ax_theta = self.fig.add_subplot(gs[0, 1])
            self.ax_theta.set_title("Angle ($\\Theta$) over Time")
            self.line_theta, = self.ax_theta.plot(self.time_data, self.theta_data, color='b')
            self.ax_theta.set_ylabel("$\\Theta$ (rad)")
            self.ax_theta.set_xlabel("Time (s)")
            self.ax_theta.grid(True)

            # dTheta plot
            self.ax_dtheta = self.fig.add_subplot(gs[1, 1])
            self.ax_dtheta.set_title("Angular Velocity ($\\dot{\\Theta}$) over Time")
            self.line_dtheta, = self.ax_dtheta.plot(self.time_data, self.dtheta_data, color='g')
            self.ax_dtheta.set_ylabel("$\\dot{\\Theta}$ (rad/s)")
            self.ax_dtheta.set_xlabel("Time (s)")
            self.ax_dtheta.grid(True)

            plt.tight_layout()
        else:
            # Update pendulum rod and bob
            self.rod.set_data(bx, by)
            self.bob.center = (bx[-1], by[-1])

            # Update title
            self.title.set_text(f"Time: {self.time_elapsed:.2f}s  $\\Theta$:{theta:.2f} rad   $\\dot{{\\Theta }}$:{dtheta:.2f} rad/s")

            # Update theta plot
            self.line_theta.set_data(self.time_data, self.theta_data)
            self.ax_theta.relim()
            self.ax_theta.autoscale_view()

            # Update dtheta plot
            self.line_dtheta.set_data(self.time_data, self.dtheta_data)
            self.ax_dtheta.relim()
            self.ax_dtheta.autoscale_view()

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
