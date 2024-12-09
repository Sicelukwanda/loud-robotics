import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

class Link:
    def __init__(
        self, 
        length, 
        fixed=False, 
        angle_limits=None, 
        fixedOrigin=False,
        circle_offsets=None,
        circle_radii=None,
        tensor_args={'dtype': torch.float32, 'device': 'cpu'}
    ):
        """
        length: float, length of the link
        fixed: bool, whether this joint angle is fixed or actuated
        angle_limits: tuple (min_angle, max_angle) or None for no limits
        fixedOrigin: bool, if True, this link's origin is at (0,0) and does not move.
        circle_offsets: list of floats specifying offsets along the link length
        circle_radii: list of floats specifying the radius for each circle
        """
        self.length = length
        self.fixed = fixed
        self.angle_limits = angle_limits
        self.fixedOrigin = fixedOrigin

        # Circle parameters: store them as tensors for convenience
        if circle_offsets is None:
            circle_offsets = []
        if circle_radii is None:
            circle_radii = []
        self.circle_offsets = torch.tensor(circle_offsets, **tensor_args)
        self.circle_radii = torch.tensor(circle_radii, **tensor_args)

        # These attributes are computed during forward kinematics:
        # self.origin and self.endpoint are handled by the arm class.
        # The circles positions are also computed by the arm class.

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
        links: list of Link objects
        dt: timestep
        tensor_args: device and dtype
        seed: random seed
        damping: damping coefficient
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.links = links
        self.tensor_args = tensor_args
        self.dt = dt
        self.damping = damping

        self.actuated_indices = [i for i, link in enumerate(links) if not link.fixed]
        self.fixed_indices = [i for i, link in enumerate(links) if link.fixed]

        self.state_dim = 2 * len(self.actuated_indices)
        self.action_dim = len(self.actuated_indices)
        self.num_particles = 1

        # Visualization parameters
        self.radius = 0.05
        self.scale = 1.0
        self.fig = None
        self.ax_arm = None
        self.ax_theta = None
        self.ax_dtheta = None
        self.title = None
        self.colors = list(mcolors.TABLEAU_COLORS.values())

        self.time_data = []
        self.theta_data = []
        self.dtheta_data = []

        # Initialize default start state 
        if starting_angle_config is None:
        # Default start angles near zero
            start_angles = torch.zeros(len(self.links), **self.tensor_args) + (torch.randn(len(self.links), **self.tensor_args)*0.1)
        else:
            assert starting_angle_config.shape == torch.Size([len(self.links)])
            start_angles = starting_angle_config

        start_velocities = torch.zeros(len(self.links), **self.tensor_args)

        self.start_angles = start_angles
        self.start_velocities = start_velocities

        plt.ion()
        self.reset()

    @property
    def init_state_distribution(self):
        mean_state = torch.cat([
            self.start_angles[self.actuated_indices],
            self.start_velocities[self.actuated_indices]
        ])
        cov = torch.eye(self.state_dim, **self.tensor_args)*0.1
        return torch.distributions.MultivariateNormal(mean_state, cov)

    def reset(self, num_particles=1):
        self.num_particles = num_particles
        sampled_state = self.init_state_distribution.sample((num_particles,))
        self.x = sampled_state
        self.time_elapsed = 0.0
        self.time_data = [self.time_elapsed]

        angles = self.x[:, :len(self.actuated_indices)]
        dangles = self.x[:, len(self.actuated_indices):]
        self.theta_data = [angles.cpu().numpy()]
        self.dtheta_data = [dangles.cpu().numpy()]

        return self.x

    def _build_full_angle_vector(self, actuated_angles):
        num_particles = actuated_angles.shape[0]
        full_angles = torch.zeros(num_particles, len(self.links), **self.tensor_args)
        a_count = 0
        for i, link in enumerate(self.links):
            if link.fixed:
                full_angles[:, i] = self.start_angles[i]
            else:
                full_angles[:, i] = actuated_angles[:, a_count]
                a_count += 1
        return full_angles

    def forward_kinematics(self, full_angles):
        """
        Returns:
          origins: (num_particles, num_links, 2)
          endpoints: (num_particles, num_links, 2)
          circle_centers: list of tensors, (num_particles, num_links, num_circles, 2)
          circle_radii: list or tensor (num_links, num_circles)
        """
        num_particles, num_links = full_angles.shape
        origins = torch.zeros(num_particles, num_links, 2, **self.tensor_args)
        endpoints = torch.zeros(num_particles, num_links, 2, **self.tensor_args)

        # Circle positions container:
        # Varying number of circles per link, we’ll store them in a list of tensors.
        # Or we can store them in a nested structure since number of circles can differ.
        # For simplicity, let's store them in a list of size num_links, each a tensor of shape (num_particles, num_circles, 2).
        circle_positions = []

        for i, link in enumerate(self.links):
            # Compute the global angle for this link:
            # The global angle is sum of all angles up to this link
            global_angle = torch.sum(full_angles[:, :i+1], dim=1)
            if link.fixedOrigin and i == 0:
                # Base link at (0,0)
                current_origin = torch.zeros(num_particles, 2, **self.tensor_args)
            else:
                # Origin of this link = endpoint of previous link
                # We must have computed endpoints for i-1
                if i == 0:
                    # If first link and not fixedOrigin, still start at (0,0)
                    # but we can imagine it's attached to something else
                    current_origin = torch.zeros(num_particles, 2, **self.tensor_args)
                else:
                    current_origin = endpoints[:, i-1, :]

            # End of this link
            dx = link.length * torch.cos(global_angle)
            dy = link.length * torch.sin(global_angle)
            current_endpoint = torch.stack([current_origin[:,0] + dx, current_origin[:,1] + dy], dim=1)

            # Store origins and endpoints
            origins[:, i, :] = current_origin
            endpoints[:, i, :] = current_endpoint

            # Compute circle positions for this link
            # For each circle offset, the circle center in local coordinates is (offset, 0)
            # In global coordinates:
            # circle_x = origin_x + offset * cos(global_angle)
            # circle_y = origin_y + offset * sin(global_angle)

            if len(link.circle_offsets) > 0:
                # Expand global_angle, current_origin for broadcasting
                ga_expanded = global_angle.unsqueeze(1)  # (num_particles, 1)
                ox = current_origin[:,0].unsqueeze(1)    # (num_particles, 1)
                oy = current_origin[:,1].unsqueeze(1)

                cox = ox + link.circle_offsets * torch.cos(ga_expanded)
                coy = oy + link.circle_offsets * torch.sin(ga_expanded)
                # cox, coy: (num_particles, num_circles)

                # stack along last dimension to get (num_particles, num_circles, 2)
                link_circle_pos = torch.stack([cox, coy], dim=2)
            else:
                # No circles on this link
                link_circle_pos = torch.zeros(num_particles, 0, 2, **self.tensor_args)

            circle_positions.append(link_circle_pos)

        return origins, endpoints, circle_positions

    def step(self, u=None):
        if u is None:
            u = torch.zeros((self.num_particles, self.action_dim), **self.tensor_args)

        angles = self.x[:, :len(self.actuated_indices)]
        dangles = self.x[:, len(self.actuated_indices):]

        # Simple dynamics
        ddangles = u - self.damping * dangles
        dangles_new = dangles + ddangles * self.dt
        angles_new = angles + dangles_new * self.dt

        full_angles = self._build_full_angle_vector(angles_new)
        # Apply angle limits
        for i, link in enumerate(self.links):
            if link.angle_limits is not None and not link.fixed:
                full_angles[:, i] = link.apply_angle_limits(full_angles[:, i])

        # Extract actuated angles again after limiting
        actuated_angles = full_angles[:, self.actuated_indices]
        self.x = torch.cat([actuated_angles, dangles_new], dim=1)

        self.time_elapsed += self.dt
        self.time_data.append(self.time_elapsed)
        self.theta_data.append(actuated_angles.cpu().numpy())
        self.dtheta_data.append(dangles_new.cpu().numpy())

        if len(self.theta_data) > 50:
            self.theta_data.pop(0)
            self.dtheta_data.pop(0)
            self.time_data.pop(0)

        return self.x

    def visualize(self, show_plot=True, show_circles=True):
        angles = self.x[:, :len(self.actuated_indices)]
        full_angles = self._build_full_angle_vector(angles)
        origins, endpoints, circle_positions = self.forward_kinematics(full_angles)

        num_particles = self.num_particles
        num_joints = len(self.actuated_indices)

        if self.fig is None:
            self.fig = plt.figure(figsize=(10,6))
            gs = self.fig.add_gridspec(2,2)

            # Arm plot (left)
            self.ax_arm = self.fig.add_subplot(gs[:,0])
            self.ax_arm.set_aspect('equal')
            arm_length = sum([l.length for l in self.links])*1.2
            self.ax_arm.set_xlim(-arm_length, arm_length)
            self.ax_arm.set_ylim(-arm_length, arm_length)
            self.ax_arm.grid(True)
            self.title = self.ax_arm.set_title(f"Time: {self.time_elapsed:.2f}s")

            # Theta plot (top-right)
            self.ax_theta = self.fig.add_subplot(gs[0, 1])
            self.ax_theta.set_title("Angles over Time (All Particles)")
            self.ax_theta.set_xlabel("Time (s)")
            self.ax_theta.set_ylabel("Angle (rad)")
            self.ax_theta.grid(True)

            # dTheta plot (bottom-right)
            self.ax_dtheta = self.fig.add_subplot(gs[1, 1])
            self.ax_dtheta.set_title("Angular Velocities over Time (All Particles)")
            self.ax_dtheta.set_xlabel("Time (s)")
            self.ax_dtheta.set_ylabel("Angular Velocity (rad/s)")
            self.ax_dtheta.grid(True)

            # We will store line references for each particle and each joint
            self.lines_theta = []
            self.lines_dtheta = []

            # Initialize line objects for all particles and joints
            # Each line corresponds to a single joint of a single particle
            for p_idx in range(num_particles):
                particle_color = self.colors[p_idx % len(self.colors)]
                particle_lines_theta = []
                particle_lines_dtheta = []
                for j in range(num_joints):
                    # Extract angle trajectory for particle p_idx and joint j
                    angle_traj = [t[p_idx, j] for t in self.theta_data]
                    (theta_line,) = self.ax_theta.plot(self.time_data, angle_traj,
                                                    label=f'Particle {p_idx}, Joint {self.actuated_indices[j]}',
                                                    color=particle_color)
                    particle_lines_theta.append(theta_line)

                    # Extract dangle trajectory for particle p_idx and joint j
                    dangle_traj = [t[p_idx, j] for t in self.dtheta_data]
                    (dtheta_line,) = self.ax_dtheta.plot(self.time_data, dangle_traj,
                                                        label=f'Particle {p_idx}, $\\dot{{q}}$ {self.actuated_indices[j]}',
                                                        color=particle_color)
                    particle_lines_dtheta.append(dtheta_line)

                self.lines_theta.append(particle_lines_theta)
                self.lines_dtheta.append(particle_lines_dtheta)

            self.ax_theta.legend()
            self.ax_dtheta.legend()

            plt.tight_layout()

        else:
            # If figure already exists, just update arm axes
            self.ax_arm.clear()
            self.ax_arm.set_aspect('equal')
            arm_length = sum([l.length for l in self.links])*1.2
            self.ax_arm.set_xlim(-arm_length, arm_length)
            self.ax_arm.set_ylim(-arm_length, arm_length)
            self.ax_arm.grid(True)
            self.title = self.ax_arm.set_title(f"Time: {self.time_elapsed:.2f}s")

        # Plot all particles and their arms
        for p_idx in range(num_particles):
            particle_color = self.colors[p_idx % len(self.colors)]
            for i, link in enumerate(self.links):
                ox, oy = origins[p_idx, i, :].cpu().numpy()
                ex, ey = endpoints[p_idx, i, :].cpu().numpy()
                self.ax_arm.plot([ox, ex], [oy, ey], linewidth=4, color=particle_color)
                self.ax_arm.add_patch(Circle((ex, ey), radius=self.radius, color=particle_color))

                # Plot circles if requested
                if show_circles and link.circle_offsets.numel() > 0:
                    for c_i in range(link.circle_offsets.shape[0]):
                        cx, cy = circle_positions[i][p_idx, c_i, :].cpu().numpy()
                        self.ax_arm.add_patch(Circle((cx, cy),
                                                    radius=link.circle_radii[c_i].item(),
                                                    fill=False, edgecolor=particle_color, linestyle='--'))

        # Update theta lines for ALL particles and joints
        for p_idx in range(num_particles):
            for j in range(num_joints):
                angle_traj = [t[p_idx, j] for t in self.theta_data]
                self.lines_theta[p_idx][j].set_data(self.time_data, angle_traj)

        # Rescale theta plot
        self.ax_theta.relim()
        self.ax_theta.autoscale_view()

        # Update dtheta lines for ALL particles and joints
        for p_idx in range(num_particles):
            for j in range(num_joints):
                dangle_traj = [t[p_idx, j] for t in self.dtheta_data]
                self.lines_dtheta[p_idx][j].set_data(self.time_data, dangle_traj)

        # Rescale dtheta plot
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
    plt.show()
