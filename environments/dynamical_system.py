import torch
from torch.distributions import MultivariateNormal
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors


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
        self.m = 2.0
        self.l = 1.0
        self.g = 9.8

        # state space: [theta, dtheta]
        self.state_dim = 2
        # action space: [u]
        self.action_dim = 1

        self.num_particles = 1

        # plotting
        self.radius = 0.1
        self.scale = 1.5
    
        self.dt = dt

        # Initialize plot attributes
        self.fig = None
        self.ax_pendulum = None
        self.ax_theta = None
        self.ax_dtheta = None
        self.rods = []
        self.bobs = []
        self.title = None

        # Data for plots
        self.time_data = []
        self.theta_data = []
        self.dtheta_data = []

        # Colors for multiple particles
        self.colors = list(mcolors.TABLEAU_COLORS.values())

        # Turn on interactive plotting
        plt.ion()
    
        # initializer
        self.reset()

    @property
    def init_state_distribution(self):
        return MultivariateNormal(
            torch.tensor([-torch.pi/2.0, 0.0], **self.tensor_args),
            torch.eye(2, **self.tensor_args)
        )

    def reset(self, num_particles=1):
        self.num_particles = num_particles
        self.x = self.init_state_distribution.sample((num_particles,))
        self.time_elapsed = 0.0
        self.time_data = [self.time_elapsed]
        self.theta_data = [self.x[:, 0].cpu().numpy()]
        self.dtheta_data = [self.x[:, 1].cpu().numpy()]
        return self.x

    def _calculate_next_state(self, theta, dtheta, u):
        """Calculate next state based on current angle, angular velocity, and input force."""
        dtheta_new = dtheta + (-3 * self.g / (2 * self.l) * torch.sin(theta + torch.pi) + 3. / (self.m * self.l**2) * u.squeeze() -0.1*dtheta) * self.dt
        theta_new = theta + dtheta_new * self.dt
        return theta_new, dtheta_new

    def step(self, u=None):
        """
        Step the simulation forward by one time step.
        - expects a control input u of shape (num_particles, 1), i.e., an action for each state particle
        - returns the new state of the system as a tensor of shape (num_particles, 2)
        """

        if u is None: # passive dynamics if no control action is given
            u = torch.zeros((self.num_particles, 1), **self.tensor_args)

        theta, dtheta = self.x[:, 0], self.x[:, 1]
        theta_new, dtheta_new = self._calculate_next_state(theta, dtheta, u)
        self.x = torch.stack([theta_new, dtheta_new], dim=1)
        self.time_elapsed += self.dt

        # Update data lists
        self.time_data.append(self.time_elapsed)
        self.theta_data.append(self.x[:, 0].cpu().numpy().flatten())
        self.dtheta_data.append(self.x[:, 1].cpu().numpy().flatten())

        if len(self.theta_data) > 50:
            self.theta_data.pop(0)
            self.dtheta_data.pop(0)
            self.time_data.pop(0)

        return self.x

    def plot_pendulum(self, theta):
        """Plot the pendulum rods and bobs based on the current angles."""
        for i in range(self.num_particles):
            bx = np.array([0.0, self.l * math.sin(-theta[i])])
            by = np.array([0.0, self.l * math.cos(-theta[i])])
            if i < len(self.rods):
                # Update existing rods and bobs
                self.rods[i].set_data(bx, by)
                self.bobs[i].center = (bx[-1], by[-1])
            else:
                # Create new rods and bobs
                rod, = self.ax_pendulum.plot(bx, by, lw=4, color=self.colors[i % len(self.colors)])
                bob = Circle((bx[-1], by[-1]), self.radius, color=self.colors[i % len(self.colors)])
                self.ax_pendulum.add_patch(bob)
                self.rods.append(rod)
                self.bobs.append(bob)

    def visualize(self, x=None, show_plot=True):
        if x is None:
            x = self.x.clone()
        theta, dtheta = x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy()  # Convert tensor to numpy array

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

            # Plot pendulum rods and bobs
            self.plot_pendulum(theta)

            # Title
            self.title = self.ax_pendulum.set_title(f"Time: {self.time_elapsed:.2f}s")

            # Theta plot
            self.ax_theta = self.fig.add_subplot(gs[0, 1])
            self.ax_theta.set_title("Angle ($\Theta$) over Time")
            self.lines_theta = []
            for i in range(self.num_particles):
                line_theta, = self.ax_theta.plot(self.time_data, [t[i].item() for t in self.theta_data], color=self.colors[i % len(self.colors)], label=f'Particle {i}')
                self.lines_theta.append(line_theta)
            self.ax_theta.set_ylabel("$\Theta$ (rad)")
            self.ax_theta.set_xlabel("Time (s)")
            self.ax_theta.grid(True)
            self.ax_theta.legend()

            # dTheta plot
            self.ax_dtheta = self.fig.add_subplot(gs[1, 1])
            self.ax_dtheta.set_title("Angular Velocity ($\dot{\Theta}$) over Time")
            self.lines_dtheta = []
            for i in range(self.num_particles):
                line_dtheta, = self.ax_dtheta.plot(self.time_data, [t[i].item() for t in self.dtheta_data], color=self.colors[i % len(self.colors)], label=f'Particle {i}')
                self.lines_dtheta.append(line_dtheta)
            self.ax_dtheta.set_ylabel("$\dot{\Theta}$ (rad/s)")
            self.ax_dtheta.set_xlabel("Time (s)")
            self.ax_dtheta.grid(True)
            self.ax_dtheta.legend()

            plt.tight_layout()
        else:
            # Update pendulum rods and bobs
            self.plot_pendulum(theta)

            # Update title
            self.title.set_text(f"Time: {self.time_elapsed:.2f}s")

            # Update theta plot
            for i in range(self.num_particles):
                self.lines_theta[i].set_data(self.time_data, [t[i].item() for t in self.theta_data])
            self.ax_theta.relim()
            self.ax_theta.autoscale_view()

            # Update dtheta plot
            for i in range(self.num_particles):
                self.lines_dtheta[i].set_data(self.time_data, [t[i].item() for t in self.dtheta_data])
            self.ax_dtheta.relim()
            self.ax_dtheta.autoscale_view()

        if show_plot:
            plt.draw()
            plt.pause(0.001)  # Pause to update the plot
