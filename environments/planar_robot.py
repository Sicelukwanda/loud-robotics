import torch
import numpy as np
import math
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

class Link:
    def __init__(self, length, fixed=False, angle_limits=None, fixedOrigin=False):
        """
        length: float, length of the link
        fixed: bool, whether this joint angle is fixed or actuated
        angle_limits: tuple (min_angle, max_angle) or None for no limits
        fixedOrigin: bool, if True, this link's origin is fixed at (0,0)
                     and does not depend on previous links.
                     Typically True for the base link.
        """
        self.length = length
        self.fixed = fixed
        self.angle_limits = angle_limits
        self.fixedOrigin = fixedOrigin

        # These attributes will be computed during forward kinematics:
        self.origin = np.array([0.0, 0.0])   # Link origin
        self.endpoint = np.array([0.0, 0.0]) # Link endpoint (x,y position)
    
    def apply_angle_limits(self, angle):
        if self.angle_limits is not None:
            min_angle, max_angle = self.angle_limits
            return torch.clamp(angle, min_angle, max_angle)
        return angle

class DPlanarRobot:
    def __init__(
        self,
        links, 
        dt=0.07, 
        tensor_args={'dtype': torch.float32, 'device': 'cpu'},
        starting_angle_config = None,
        seed=0,
        damping=0.1
    ):
        """
        links: list of Link objects defining the chain
        dt: timestep
        tensor_args: device and dtype
        seed: random seed
        damping: damping coefficient for the joint dynamics
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.links = links
        self.tensor_args = tensor_args
        self.dt = dt
        self.damping = damping

        # Identify which joints are actuated (not fixed)
        self.actuated_indices = [i for i, link in enumerate(links) if not link.fixed]
        self.fixed_indices = [i for i, link in enumerate(links) if link.fixed]

        # State dimension:
        # For each actuated joint we have angle and angular velocity
        self.state_dim = 2 * len(self.actuated_indices)

        # Action dimension:
        # For each actuated joint we have a torque input (or control input u)
        self.action_dim = len(self.actuated_indices)

        # Number of particles (parallel simulations)
        self.num_particles = 1

        # Visualization parameters
        self.radius = 0.05
        self.scale = 1.0
        self.fig = None
        self.ax_arm = None
        self.ax_theta = None
        self.ax_dtheta = None
        self.lines_theta = []
        self.lines_dtheta = []
        self.title = None
        self.colors = list(mcolors.TABLEAU_COLORS.values())

        # Data logging
        self.time_data = []
        self.theta_data = []
        self.dtheta_data = []

        # Turn on interactive plotting
        plt.ion()

        # Initialize default start state 
        if starting_angle_config is None:
            # initialize randomly with values close to zero
            # We need to form a mean state: angles and velocities
            start_angles = torch.zeros(len(self.links), **self.tensor_args)
            # For fixed links, angles are not part of the state. But we still keep track of them.
            # Just randomize slightly around zero for demonstration.
            start_angles += (torch.randn(len(self.links), **self.tensor_args)*0.1)

            # Set a baseline initial state
            self.start_angles = start_angles
        else:
            assert starting_angle_config.shape == torch.Size([len(self.links)])
            self.start_angles = starting_angle_config
            
        
        self.start_velocities = torch.zeros(len(self.links), **self.tensor_args)

        self.reset()
    
    @property
    def init_state_distribution(self):
        # Only for actuated joints:
        mean_state = torch.cat([
            self.start_angles[self.actuated_indices],
            self.start_velocities[self.actuated_indices]
        ])
        cov = torch.eye(self.state_dim, **self.tensor_args)*0.1
        return MultivariateNormal(mean_state, cov)

    def reset(self, num_particles=1):
        self.num_particles = num_particles
        # Sample initial states
        sampled_state = self.init_state_distribution.sample((num_particles,))
        # State includes only actuated joints
        self.x = sampled_state  # shape: (num_particles, state_dim)
        
        # Keep track of time
        self.time_elapsed = 0.0
        self.time_data = [self.time_elapsed]

        # Extract angles and velocities for logging
        angles = self.x[:, :len(self.actuated_indices)] # actuated angles
        dangles = self.x[:, len(self.actuated_indices):]
        self.theta_data = [angles.cpu().numpy()]
        self.dtheta_data = [dangles.cpu().numpy()]

        return self.x

    def forward_kinematics(self, angles_all):
        """
        Compute the forward kinematics for each particle.
        angles_all: (num_particles, num_links) tensor of all joint angles (including fixed ones).
        Returns:
          origins: (num_particles, num_links, 2)
          endpoints: (num_particles, num_links, 2)
        """
        num_particles, num_links = angles_all.shape
        origins = torch.zeros(num_particles, num_links, 2, **self.tensor_args)
        endpoints = torch.zeros(num_particles, num_links, 2, **self.tensor_args)

        for p in range(num_particles):
            # Compute cumulative angles and positions
            current_pos = torch.tensor([0.0,0.0], **self.tensor_args)
            current_angle = 0.0
            for i, link in enumerate(self.links):
                # If this link has a fixedOrigin, reset current_pos and current_angle?
                # Typically only the first link might have fixedOrigin=True
                if link.fixedOrigin and i == 0:
                    current_pos = torch.tensor([0.0,0.0], **self.tensor_args)
                    current_angle = 0.0

                # Add the angle of this link to current_angle
                current_angle += angles_all[p, i]
                origins[p, i, :] = current_pos

                # Compute endpoint of this link
                end_x = current_pos[0] + link.length * torch.cos(current_angle)
                end_y = current_pos[1] + link.length * torch.sin(current_angle)
                endpoints[p, i, :] = torch.stack([end_x, end_y])

                # The next link origin is the endpoint of the current link
                current_pos = endpoints[p, i, :]

        return origins, endpoints

    def _apply_angle_limits(self, angles_all):
        """
        angles_all: (num_particles, num_links)
        Applies angle limits to each link if provided.
        """
        for i, link in enumerate(self.links):
            if link.angle_limits is not None:
                min_a, max_a = link.angle_limits
                angles_all[:, i] = torch.clamp(angles_all[:, i], min_a, max_a)
        return angles_all

    def step(self, u=None):
        """
        Step the simulation forward by one time step.
        u: (num_particles, action_dim) tensor of torques for actuated joints.
           If None, use zero torques (passive dynamics).
        """
        if u is None:
            u = torch.zeros((self.num_particles, self.action_dim), **self.tensor_args)

        # Extract angles and velocities of actuated joints
        angles = self.x[:, :len(self.actuated_indices)]
        dangles = self.x[:, len(self.actuated_indices):]

        # Simple dynamics:
        # ddangles = u - damping*dangles
        ddangles = u - self.damping * dangles

        # Update velocities and angles
        dangles_new = dangles + ddangles * self.dt
        angles_new = angles + dangles_new * self.dt

        # Recombine fixed and actuated angles for limit checking
        # We have full num_links = fixed + actuated
        full_angles = self._build_full_angle_vector(angles_new)
        # Apply angle limits if any
        full_angles = self._apply_angle_limits(full_angles)
        
        # Now extract back the actuated joint angles after limit enforcement
        # Because limits might have changed actuated angles as well
        angles_clamped = full_angles[:, self.actuated_indices]
        
        # Update state
        self.x = torch.cat([angles_clamped, dangles_new], dim=1)
        
        self.time_elapsed += self.dt
        self.time_data.append(self.time_elapsed)

        # Log data
        self.theta_data.append(angles_clamped.cpu().numpy())
        self.dtheta_data.append(dangles_new.cpu().numpy())

        # Limit memory
        if len(self.theta_data) > 50:
            self.theta_data.pop(0)
            self.dtheta_data.pop(0)
            self.time_data.pop(0)

        return self.x

    def _build_full_angle_vector(self, actuated_angles):
        """
        Build the full angle vector (including fixed joints) given only actuated angles.
        actuated_angles: (num_particles, #actuated_joints)
        Returns:
          full_angles: (num_particles, num_links) with both fixed and actuated angles.
        """
        num_particles = actuated_angles.shape[0]
        full_angles = torch.zeros(num_particles, len(self.links), **self.tensor_args)

        # We'll have counters to fill in angles
        # For fixed joints, we just use the start_angles as their angle.
        # For actuated joints, we use the angles from actuated_angles
        a_count = 0
        for i, link in enumerate(self.links):
            if link.fixed:
                # fixed angle does not change from initial (or can be set to a constant)
                # Here we just keep it at the initially set angle for simplicity
                # You might store fixed angles separately if desired
                full_angles[:, i] = self.start_angles[i]  # same angle for all particles
            else:
                full_angles[:, i] = actuated_angles[:, a_count]
                a_count += 1
        return full_angles

    def visualize(self, show_plot=True):
        # Reconstruct full angles from current state for plotting
        angles = self.x[:, :len(self.actuated_indices)]
        full_angles = self._build_full_angle_vector(angles)

        # Compute forward kinematics for all particles
        origins, endpoints = self.forward_kinematics(full_angles)

        num_particles = self.num_particles
        num_joints = len(self.actuated_indices)

        # For plotting, we'll just show the first particle to avoid complications
        # (You can extend this logic to handle multiple particles if desired.)
        p_idx = 0  # first particle

        if self.fig is None:
            self.fig = plt.figure(figsize=(10,6))
            gs = self.fig.add_gridspec(2,2)

            self.ax_arm = self.fig.add_subplot(gs[:,0])
            self.ax_arm.set_aspect('equal')
            arm_length = sum([l.length for l in self.links])*1.2
            self.ax_arm.set_xlim(-arm_length, arm_length)
            self.ax_arm.set_ylim(-arm_length, arm_length)
            self.ax_arm.grid(True)
            self.title = self.ax_arm.set_title(f"Time: {self.time_elapsed:.2f}s")

            # Plot the robot arm (for the first particle)
            for i, link in enumerate(self.links):
                ox, oy = origins[p_idx, i, :].cpu().numpy()
                ex, ey = endpoints[p_idx, i, :].cpu().numpy()
                self.ax_arm.plot([ox, ex], [oy, ey], linewidth=4, color=self.colors[i % len(self.colors)])
                self.ax_arm.add_patch(Circle((ex, ey), radius=self.radius, color=self.colors[i % len(self.colors)]))

            # Theta plot
            self.ax_theta = self.fig.add_subplot(gs[0, 1])
            self.ax_theta.set_title("Angles over Time")
            self.ax_theta.set_xlabel("Time (s)")
            self.ax_theta.set_ylabel("Angle (rad)")
            self.ax_theta.grid(True)

            self.lines_theta = []
            for i in range(num_joints):
                # Extract the angle trajectory for joint i, first particle:
                angle_traj = [t[p_idx, i] for t in self.theta_data]
                (line,) = self.ax_theta.plot(self.time_data, angle_traj, 
                                            label=f'Joint {self.actuated_indices[i]}', 
                                            color=self.colors[i % len(self.colors)])
                self.lines_theta.append(line)
            self.ax_theta.legend()

            # dTheta plot
            self.ax_dtheta = self.fig.add_subplot(gs[1, 1])
            self.ax_dtheta.set_title("Angular Velocities over Time")
            self.ax_dtheta.set_xlabel("Time (s)")
            self.ax_dtheta.set_ylabel("Angular Velocity (rad/s)")
            self.ax_dtheta.grid(True)

            self.lines_dtheta = []
            for i in range(num_joints):
                dangle_traj = [t[p_idx, i] for t in self.dtheta_data]
                (line,) = self.ax_dtheta.plot(self.time_data, dangle_traj, 
                                            label=f'Joint {self.actuated_indices[i]}', 
                                            color=self.colors[i % len(self.colors)])
                self.lines_dtheta.append(line)
            self.ax_dtheta.legend()

            plt.tight_layout()

        else:
            # Update arm plot
            self.ax_arm.clear()
            self.ax_arm.set_aspect('equal')
            arm_length = sum([l.length for l in self.links])*1.2
            self.ax_arm.set_xlim(-arm_length, arm_length)
            self.ax_arm.set_ylim(-arm_length, arm_length)
            self.ax_arm.grid(True)
            self.title = self.ax_arm.set_title(f"Time: {self.time_elapsed:.2f}s")

            # Re-draw the arm for the first particle
            for i, link in enumerate(self.links):
                ox, oy = origins[p_idx, i, :].cpu().numpy()
                ex, ey = endpoints[p_idx, i, :].cpu().numpy()
                self.ax_arm.plot([ox, ex], [oy, ey], linewidth=4, color=self.colors[i % len(self.colors)])
                self.ax_arm.add_patch(Circle((ex, ey), radius=self.radius, color=self.colors[i % len(self.colors)]))

            # Update theta lines
            for i in range(num_joints):
                angle_traj = [t[p_idx, i] for t in self.theta_data]
                self.lines_theta[i].set_data(self.time_data, angle_traj)
            self.ax_theta.relim()
            self.ax_theta.autoscale_view()

            # Update dtheta lines
            for i in range(num_joints):
                dangle_traj = [t[p_idx, i] for t in self.dtheta_data]
                self.lines_dtheta[i].set_data(self.time_data, dangle_traj)
            self.ax_dtheta.relim()
            self.ax_dtheta.autoscale_view()

        if show_plot:
            plt.draw()
            plt.pause(0.001)


# Example usage:
# Define a 3-link arm: first link fixedOrigin and actuated, second link fixed, third link actuated
# Just as a demonstration:
if __name__ == "__main__":
    tensor_args = {"dtype": torch.float32, "device": "cpu"}

    links = [
        Link(length=1.0, fixed=False, angle_limits=(-math.pi, math.pi), fixedOrigin=True),
        Link(length=0.5, fixed=True),  # a fixed joint angle link
        Link(length=0.5, fixed=False, angle_limits=(-math.pi/2, math.pi/2))
    ]

    start_angles = torch.zeros(len(links), **tensor_args)
    env = DPlanarRobot(links=links, dt=0.05, tensor_args=tensor_args, starting_angle_config=start_angles, seed=0)
    x_init = env.reset(num_particles=3)  # three parallel arms
    print(f"starting states has shape:{x_init.shape} and values \n{x_init}")
    for t in range(50):
        u = torch.zeros((env.num_particles, env.action_dim), **tensor_args)
        env.step(u)
        env.visualize(show_plot=True)
    plt.show()
