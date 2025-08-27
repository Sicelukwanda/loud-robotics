import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors

from .dynamics_utils import circle_sdf

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

class CircleObstacle:
    def __init__(self, center, radius, tensor_args={'dtype': torch.float32, 'device': 'cpu'}, color='red'):
        """
        center: tuple (x, y) - center of the circle obstacle
        radius: float - radius of the circle obstacle
        color: str - color for visualization
        """
        self.center = torch.tensor(center, **tensor_args)
        self.radius = torch.tensor(radius, **tensor_args)
        self.tensor_args = tensor_args
        self.color = color
    
    def sdf(self, points):
        """
        Compute SDF for points with respect to this circle obstacle.
        
        Args:
            points: (N, 2) tensor of query points
            
        Returns:
            sdf_values: (N,) tensor - negative inside obstacle, positive outside
        """
        # Distance from points to circle center
        diff = points - self.center.unsqueeze(0)  # (N, 2)
        distances = torch.sqrt((diff**2).sum(dim=1))  # (N,)
        return distances - self.radius

class RectangleObstacle:
    def __init__(self, center, width, height, angle=0.0, tensor_args={'dtype': torch.float32, 'device': 'cpu'}, color='red'):
        """
        center: tuple (x, y) - center of the rectangle
        width: float - width of the rectangle  
        height: float - height of the rectangle  
        angle: float - rotation angle in radians (default 0)
        color: str - color for visualization
        """
        self.center = torch.tensor(center, **tensor_args)
        self.width = torch.tensor(width, **tensor_args)
        self.height = torch.tensor(height, **tensor_args)
        self.angle = torch.tensor(angle, **tensor_args)
        self.tensor_args = tensor_args
        self.color = color
        
        # Precompute rotation matrix
        cos_a = torch.cos(self.angle)
        sin_a = torch.sin(self.angle)
        self.rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], **tensor_args)
        
        # Precompute rotated corners for visualization
        self._compute_rotated_corners()
    
    def _compute_rotated_corners(self):
        """Precompute the rotated corner positions for efficient visualization."""
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Corner positions in local coordinate system (relative to center)
        corners_local = torch.tensor([
            [-half_width, -half_height],  # bottom-left
            [half_width, -half_height],   # bottom-right
            [half_width, half_height],    # top-right
            [-half_width, half_height]    # top-left
        ], **self.tensor_args)
        
        # Apply rotation and translate to world coordinates
        self.corners = (corners_local @ self.rotation_matrix.T) + self.center.unsqueeze(0)
    
    def get_matplotlib_bottom_left(self):
        """Get the bottom-left corner position for matplotlib Rectangle.
        
        For matplotlib Rectangle, we need the corner that would be bottom-left
        in the rectangle's local coordinate system BEFORE rotation is applied.
        This is always the corner at (-width/2, -height/2) in local coords.
        """
        # Always use the first corner, which is defined as bottom-left in local coords
        return self.corners[0].cpu().numpy()
    
    def get_corners_numpy(self):
        """Get all corners as numpy array for plotting."""
        return self.corners.cpu().numpy()
    
    def sdf(self, points):
        """
        Compute SDF for points with respect to this rectangle obstacle.
        
        Args:
            points: (N, 2) tensor of query points
            
        Returns:
            sdf_values: (N,) tensor - negative inside obstacle, positive outside
        """
        # Transform points to rectangle's local coordinate system
        local_points = (points - self.center.unsqueeze(0)) @ self.rotation_matrix  # (N, 2)
        
        # Compute distance to rectangle in local coordinates
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Distance to each edge
        dx = torch.abs(local_points[:, 0]) - half_width
        dy = torch.abs(local_points[:, 1]) - half_height
        
        # SDF computation
        # Outside: max(dx, dy) when both dx,dy > 0, otherwise max of positive components
        # Inside: max(dx, dy) when both dx,dy < 0
        
        # Clamp to get exterior distance components
        dx_pos = torch.clamp(dx, min=0)
        dy_pos = torch.clamp(dy, min=0)
        
        # Distance to rectangle boundary
        exterior_dist = torch.sqrt(dx_pos**2 + dy_pos**2)
        interior_dist = torch.max(dx, dy)
        
        # If point is outside in any dimension, use exterior distance
        # If point is inside in both dimensions, use interior distance (negative)
        is_outside = (dx > 0) | (dy > 0)
        sdf_values = torch.where(is_outside, exterior_dist, interior_dist)
        
        return sdf_values

class DPlanarRobot:
    def __init__(
        self,
        links, 
        dt=0.07, 
        tensor_args={'dtype': torch.float32, 'device': 'cpu'},
        starting_angle_config = None,
        seed=0,
        damping=0.1,
        obstacles=None
    ):
        """
        links: list of Link objects
        dt: timestep
        tensor_args: device and dtype
        seed: random seed
        damping: damping coefficient
        obstacles: list of obstacle objects (CircleObstacle or RectangleObstacle)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.links = links
        self.tensor_args = tensor_args
        self.dt = dt
        self.damping = damping
        self.obstacles = obstacles if obstacles is not None else []

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

        # Plot obstacles first
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center = obstacle.center.cpu().numpy()
                radius = obstacle.radius.item()
                circle = Circle(center, radius, fill=True, color=obstacle.color, alpha=0.3, 
                              edgecolor=obstacle.color, linewidth=2)
                self.ax_arm.add_patch(circle)
            elif isinstance(obstacle, RectangleObstacle):
                center = obstacle.center.cpu().numpy()
                width = obstacle.width.item()
                height = obstacle.height.item()
                angle_deg = np.degrees(obstacle.angle.item())
                
                # Use precomputed rotated bottom-left corner
                bottom_left = obstacle.get_matplotlib_bottom_left()
                
                rect = Rectangle(bottom_left, width, height, angle=angle_deg, 
                               fill=True, color=obstacle.color, alpha=0.3, 
                               edgecolor=obstacle.color, linewidth=2)
                self.ax_arm.add_patch(rect)

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

    def sdf_at_points(self, points):
        """
        Compute the SDF of the robot (approximated by circles) at given query points.
        The robot may have multiple particles (num_particles states).
        We want an output of shape (num_particles, N) where N is number of points.

        Inputs:
            points: (N, 2) tensor of points at which we want the SDF

        Returns:
            sdf_values: (num_particles, N) tensor of SDF values for each particle's configuration.
        """
        # Ensure points is on correct device/dtype
        points = points.to(**self.tensor_args)

        # First, get the full angles and forward kinematics to obtain circle positions
        angles = self.x[:, :len(self.actuated_indices)]  # (num_particles, #actuated_joints)
        full_angles = self._build_full_angle_vector(angles)
        origins, endpoints, circle_positions = self.forward_kinematics(full_angles)
        # circle_positions is a list of length num_links
        # each element is (num_particles, num_circles, 2)

        num_particles = self.num_particles
        N = points.shape[0]

        # We'll collect all circles from all links into a single structure for each particle
        # to compute the union SDF. Since links may have different number of circles,
        # we concatenate them.
        all_circle_centers = []
        all_circle_radii = []

        for i, link in enumerate(self.links):
            if link.circle_offsets.numel() > 0:
                # circle_positions[i]: (num_particles, num_circles, 2)
                # circle_radii: (num_circles,)
                # We have different circles per link, but same radii across particles
                # Just replicate the radii if needed or store once
                # We'll handle each particle separately
                # Let's store these link circles and radii for later computation
                all_circle_centers.append(circle_positions[i])  # (num_particles, num_circles, 2)
                all_circle_radii.append(link.circle_radii)      # (num_circles,)

        # If no circles defined at all, the SDF might default to something large
        # or we can just return a large positive value indicating no shape.
        if len(all_circle_centers) == 0:
            # no circles -> no robot volume -> SDF is large positive
            return torch.full((num_particles, N), float('inf'), **self.tensor_args)

        # Concatenate circles from all links
        # Each all_circle_centers[i]: (num_particles, C_i, 2)
        # We want to have (num_particles, total_circles, 2)
        circle_centers_concat = torch.cat(all_circle_centers, dim=1)  # (num_particles, total_circles, 2)
        # Similarly for radii
        circle_radii_concat = torch.cat(all_circle_radii)  # (total_circles,)

        # Now we have all circles in one big tensor.
        total_circles = circle_radii_concat.shape[0]

        # We need to compute SDF for each particle and each point.
        # points: (N, 2)
        # circle_centers_concat: (num_particles, total_circles, 2)
        # circle_radii_concat: (total_circles,)

        # We'll do this for each particle:
        sdf_values = torch.empty(num_particles, N, **self.tensor_args)

        for p_idx in range(num_particles):
            c_centers = circle_centers_concat[p_idx]  # (total_circles, 2)
            # Compute SDF for all points for this particle
            # Use the circle_sdf function
            # circle_sdf expects (N,2), (M,2), (M,) and returns (N,)
            sdf_p = circle_sdf(points, c_centers, circle_radii_concat)
            sdf_values[p_idx, :] = sdf_p

        return sdf_values

    def environment_sdf_at_points(self, points):
        """
        Compute the SDF of the environment (obstacles only, without the robot) at given query points.
        
        Args:
            points: (N, 2) tensor of query points
            
        Returns:
            sdf_values: (N,) tensor of SDF values for the environment
        """
        # Ensure points is on correct device/dtype
        points = points.to(**self.tensor_args)
        N = points.shape[0]
        
        if len(self.obstacles) == 0:
            # No obstacles, return large positive values (free space)
            return torch.full((N,), float('inf'), **self.tensor_args)
        
        # Compute SDF for each obstacle and take minimum (union of obstacles)
        obstacle_sdfs = []
        for obstacle in self.obstacles:
            obstacle_sdf = obstacle.sdf(points)
            obstacle_sdfs.append(obstacle_sdf)
        
        # Stack and take minimum for union
        obstacle_sdfs_tensor = torch.stack(obstacle_sdfs, dim=1)  # (N, num_obstacles)
        environment_sdf, _ = torch.min(obstacle_sdfs_tensor, dim=1)  # (N,)
        
        return environment_sdf

    def add_obstacle(self, obstacle):
        """
        Add an obstacle to the environment.
        
        Args:
            obstacle: CircleObstacle or RectangleObstacle instance
        """
        self.obstacles.append(obstacle)

    def clear_obstacles(self):
        """
        Remove all obstacles from the environment.
        """
        self.obstacles.clear()

    def plot_sdf(self, resolution=20, sdf_min=None, sdf_max=None, particle_color=None):
        """
        Plot the SDF of each robot particle on a separate subplot.
        Uses the given resolution to define a grid of points.
        """
        # Compute bounding box for plotting
        arm_length = sum([l.length for l in self.links])*1.2
        lower = -arm_length
        upper = arm_length

        # Generate a grid of points
        xs = torch.linspace(lower, upper, resolution, **self.tensor_args)
        ys = torch.linspace(lower, upper, resolution, **self.tensor_args)

        # Create a meshgrid
        # X[i,j] = x coordinate, Y[i,j] = y coordinate
        # shape: (resolution, resolution)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')  
        # indexing='ij' means X is row-like i index and Y is column-like j index
        # We'll have to remember how to orient these when plotting.

        # Flatten the grid to pass to sdf_at_points
        points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (resolution^2, 2)

        # Compute the SDF at these points for all particles
        # returns (num_particles, resolution^2)
        sdf_values = self.sdf_at_points(points)

        num_particles = self.num_particles

        # Reshape sdf_values to (num_particles, resolution, resolution)
        sdf_values = sdf_values.view(num_particles, resolution, resolution)

        # Create a figure with one subplot per particle
        fig, axs = plt.subplots(1, num_particles, figsize=(6*num_particles,6), squeeze=False)
        axs = axs[0]  # squeeze=False returns a 2D array, we know it's 1 row

        # We'll need forward kinematics again to plot the robot configuration
        angles = self.x[:, :len(self.actuated_indices)]
        full_angles = self._build_full_angle_vector(angles)
        origins, endpoints, circle_positions = self.forward_kinematics(full_angles)

        for p_idx in range(num_particles):
            ax = axs[p_idx]
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect('equal')

            # Plot the SDF as a heatmap
            # Note: imshow expects data[y, x] indexing, so we pass sdf_values[p_idx] in normal order.
            # By default, y=0 will be top in imshow, so we set origin='lower'
            # Also define extent to map array indices to coordinates
            extent = (lower, upper, lower, upper)
            im = ax.imshow(
                sdf_values[p_idx].cpu().numpy().T,  # Transpose so that indexing matches X,Y
                extent=extent,
                origin='lower',
                cmap='jet',
                vmin=sdf_min,  # slight penetration inside the circle
                vmax=sdf_max,  # slight space outside the circle
                alpha=0.6
            )
            
            fig.colorbar(im, ax=ax, label='SDF Distance')

            # Overlay the robot arm for particle p_idx
            if particle_color is None:
                particle_color = self.colors[p_idx % len(self.colors)]

            for i, link in enumerate(self.links):
                ox, oy = origins[p_idx, i, :].cpu().numpy()
                ex, ey = endpoints[p_idx, i, :].cpu().numpy()
                # Plot the link as a line
                ax.plot([ox, ex], [oy, ey], linewidth=4, color=particle_color)
                # Add a circle at the endpoint
                ax.add_patch(Circle((ex, ey), radius=self.radius, color=particle_color))

                # If the link has offset circles, plot them
                if link.circle_offsets.numel() > 0:
                    for c_i in range(link.circle_offsets.shape[0]):
                        cx, cy = circle_positions[i][p_idx, c_i, :].cpu().numpy()
                        ax.add_patch(Circle((cx, cy),
                                            radius=link.circle_radii[c_i].item(),
                                            fill=False, edgecolor=particle_color, linestyle='--'))

            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.grid(True)

        plt.tight_layout()
        plt.draw()
        return fig


    def plot_robot_state(self, color_override=None):
        """
        Plot the SDF of each robot particle on a separate subplot.
        Uses the given resolution to define a grid of points.
        """
        # Compute bounding box for plotting
        arm_length = sum([l.length for l in self.links])*1.2
        lower = -arm_length
        upper = arm_length


        num_particles = self.num_particles

        # Create a figure with one subplot per particle
        fig, axs = plt.subplots(1, num_particles, figsize=(6*num_particles,6), squeeze=False)
        axs = axs[0]  # squeeze=False returns a 2D array, we know it's 1 row

        # We'll need forward kinematics again to plot the robot configuration
        angles = self.x[:, :len(self.actuated_indices)]
        full_angles = self._build_full_angle_vector(angles)
        origins, endpoints, circle_positions = self.forward_kinematics(full_angles)

        for p_idx in range(num_particles):
            ax = axs[p_idx]
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect('equal')

            

            for i, link in enumerate(self.links):
                # plot the robot arm for particle p_idx
                if color_override is None:
                    link_color = self.colors[i % len(self.colors)]
                else:
                    link_color = color_override

                ox, oy = origins[p_idx, i, :].cpu().numpy()
                ex, ey = endpoints[p_idx, i, :].cpu().numpy()
                # Plot the link as a line
                ax.plot([ox, ex], [oy, ey], linewidth=4, color=link_color)
                # Add a circle at the endpoint
                ax.add_patch(Circle((ex, ey), radius=self.radius, color=link_color))

                # If the link has offset circles, plot them
                if link.circle_offsets.numel() > 0:
                    for c_i in range(link.circle_offsets.shape[0]):
                        cx, cy = circle_positions[i][p_idx, c_i, :].cpu().numpy()
                        ax.add_patch(Circle((cx, cy),
                                            radius=link.circle_radii[c_i].item(),
                                            fill=False, edgecolor=link_color, linestyle='--'))

            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.grid(True)

        plt.tight_layout()
        plt.draw()
        return fig

    def plot_environment_sdf(self, resolution=100, sdf_min=-1.0, sdf_max=1.0, xlim=None, ylim=None):
        """
        Plot the SDF of the environment (obstacles only) on a 2D grid.
        
        Args:
            resolution: Grid resolution for SDF visualization
            sdf_min: Minimum SDF value for colormap
            sdf_max: Maximum SDF value for colormap
            xlim: tuple (xmin, xmax) for custom x-axis limits, or None for default
            ylim: tuple (ymin, ymax) for custom y-axis limits, or None for default
            
        Returns:
            fig: matplotlib figure
        """
        # Compute bounding box for plotting
        if xlim is None or ylim is None:
            arm_length = sum([l.length for l in self.links])*1.5
            default_lower = -arm_length
            default_upper = arm_length
        
        # Set actual plot bounds
        if xlim is not None:
            x_lower, x_upper = xlim
        else:
            x_lower, x_upper = default_lower, default_upper
            
        if ylim is not None:
            y_lower, y_upper = ylim
        else:
            y_lower, y_upper = default_lower, default_upper

        # Generate a grid of points covering the desired view area
        xs = torch.linspace(x_lower, x_upper, resolution, **self.tensor_args)
        ys = torch.linspace(y_lower, y_upper, resolution, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        # Compute environment SDF
        env_sdf = self.environment_sdf_at_points(points)
        env_sdf = env_sdf.view(resolution, resolution)

        # Create figure with proper aspect ratio for the desired view
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        
        # Set the axis limits to the desired view
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)

        # Plot SDF as heatmap
        extent = (x_lower, x_upper, y_lower, y_upper)
        im = ax.imshow(
            env_sdf.cpu().numpy().T,
            extent=extent,
            origin='lower',
            cmap='RdYlBu',
            vmin=sdf_min,
            vmax=sdf_max,
            alpha=0.8
        )
        
        # Create colorbar with proper sizing using make_axes_locatable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label='SDF Distance')

        # Overlay obstacles
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center = obstacle.center.cpu().numpy()
                radius = obstacle.radius.item()
                circle = Circle(center, radius, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(circle)
            elif isinstance(obstacle, RectangleObstacle):
                center = obstacle.center.cpu().numpy()
                width = obstacle.width.item()
                height = obstacle.height.item()
                angle_deg = np.degrees(obstacle.angle.item())
                
                # Use precomputed rotated bottom-left corner
                bottom_left = obstacle.get_matplotlib_bottom_left()
                
                rect = Rectangle(bottom_left, width, height, angle=angle_deg, 
                               fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        return fig

    def compute_configuration_space(self, joint_ranges, resolution=50):
        """
        Compute the configuration space (C-space) for the robot with obstacles.
        
        Args:
            joint_ranges: List of tuples [(min1, max1), (min2, max2), ...] for each actuated joint
            resolution: Number of samples per joint dimension
            
        Returns:
            config_grid: Grid of joint configurations
            collision_mask: Boolean mask indicating collisions (True = collision)
            sdf_values: Minimum SDF values for each configuration
        """
        num_joints = len(self.actuated_indices)
        
        if len(joint_ranges) != num_joints:
            raise ValueError(f"Expected {num_joints} joint ranges, got {len(joint_ranges)}")
        
        # Create grid of joint configurations
        joint_grids = []
        for min_val, max_val in joint_ranges:
            joint_vals = torch.linspace(min_val, max_val, resolution, **self.tensor_args)
            joint_grids.append(joint_vals)
        
        # Create meshgrid for all joint combinations
        if num_joints == 2:
            J1, J2 = torch.meshgrid(joint_grids[0], joint_grids[1], indexing='ij')
            config_grid = torch.stack([J1.flatten(), J2.flatten()], dim=1)
        elif num_joints == 3:
            J1, J2, J3 = torch.meshgrid(joint_grids[0], joint_grids[1], joint_grids[2], indexing='ij')
            config_grid = torch.stack([J1.flatten(), J2.flatten(), J3.flatten()], dim=1)
        else:
            # For higher dimensions, use manual grid construction
            import itertools
            configs = []
            for config in itertools.product(*[grid.cpu().numpy() for grid in joint_grids]):
                configs.append(config)
            config_grid = torch.tensor(configs, **self.tensor_args)
        
        num_configs = config_grid.shape[0]
        collision_mask = torch.zeros(num_configs, dtype=torch.bool, device=self.tensor_args['device'])
        sdf_values = torch.full((num_configs,), float('inf'), **self.tensor_args)
        
        # Check each configuration for collisions
        print(f"Computing C-space for {num_configs} configurations...")
        
        # Store original state
        original_x = self.x.clone()
        
        # Process configurations in batches for efficiency
        batch_size = min(100, num_configs)
        for batch_start in range(0, num_configs, batch_size):
            batch_end = min(batch_start + batch_size, num_configs)
            batch_configs = config_grid[batch_start:batch_end]
            batch_size_actual = batch_configs.shape[0]
            
            # Set robot configurations (position only, zero velocity)
            batch_states = torch.cat([
                batch_configs, 
                torch.zeros(batch_size_actual, num_joints, **self.tensor_args)
            ], dim=1)
            
            # Temporarily set robot state for this batch
            self.x = batch_states
            self.num_particles = batch_size_actual
            
            # Compute robot SDF for this batch
            if len(self.obstacles) > 0:
                # Create a grid of points around the robot workspace for collision checking
                arm_length = sum([l.length for l in self.links])
                test_points = self._create_workspace_test_points(arm_length, resolution=40, adaptive=True)
                
                # Get robot SDF at test points
                robot_sdf = self.sdf_at_points(test_points)  # (batch_size, num_test_points)
                env_sdf = self.environment_sdf_at_points(test_points)  # (num_test_points,)
                
                # Check for collisions: robot SDF < 0 AND environment SDF < 0
                # This means the point is inside both robot and obstacle
                robot_inside = robot_sdf < 0  # (batch_size, num_test_points)
                env_inside = env_sdf < 0  # (num_test_points,)
                
                # Collision occurs if any test point is inside both robot and obstacle
                collision_points = robot_inside & env_inside.unsqueeze(0)  # (batch_size, num_test_points)
                batch_collisions = torch.any(collision_points, dim=1)  # (batch_size,)
                
                # Compute minimum clearance (minimum distance between robot and obstacles)
                robot_boundary = robot_sdf <= 0.05  # Points near robot surface
                min_clearance = torch.full((batch_size_actual,), float('inf'), **self.tensor_args)
                
                for i in range(batch_size_actual):
                    robot_points = test_points[robot_boundary[i]]  # Points near robot i
                    if robot_points.numel() > 0:
                        clearances = self.environment_sdf_at_points(robot_points)
                        min_clearance[i] = torch.min(clearances)
                
                collision_mask[batch_start:batch_end] = batch_collisions
                sdf_values[batch_start:batch_end] = min_clearance
            else:
                # No obstacles - no collisions
                sdf_values[batch_start:batch_end] = float('inf')
            
            if (batch_start // batch_size) % 10 == 0:
                progress = (batch_end / num_configs) * 100
                print(f"  Progress: {progress:.1f}%")
        
        # Restore original state
        self.x = original_x
        self.num_particles = original_x.shape[0]
        
        print(f"C-space computation complete. Found {torch.sum(collision_mask).item()} collision configurations.")
        
        return config_grid, collision_mask, sdf_values

    def compute_configuration_space_per_obstacle(self, joint_ranges, resolution=50):
        """
        Compute configuration space showing individual obstacle contributions.
        
        Args:
            joint_ranges: List of (min, max) tuples for each joint
            resolution: Resolution for each joint dimension
            
        Returns:
            config_grid: (N, num_joints) tensor of all configurations
            obstacle_collision_masks: List of (N,) boolean tensors, one per obstacle
            combined_collision_mask: (N,) boolean tensor of combined collisions
            sdf_values: (N,) tensor of minimum SDF values across all obstacles
        """
        if len(joint_ranges) != 2:
            raise ValueError("Currently only supports 2-DOF robots")
        
        print(f"Computing per-obstacle C-space for {len(self.obstacles)} obstacles...")
        
        # Create configuration grid
        joint1_range, joint2_range = joint_ranges
        joint1_vals = torch.linspace(joint1_range[0], joint1_range[1], resolution, **self.tensor_args)
        joint2_vals = torch.linspace(joint2_range[0], joint2_range[1], resolution, **self.tensor_args)
        
        joint1_grid, joint2_grid = torch.meshgrid(joint1_vals, joint2_vals, indexing='ij')
        config_grid = torch.stack([joint1_grid.flatten(), joint2_grid.flatten()], dim=1)
        
        N = config_grid.shape[0]
        obstacle_collision_masks = []
        
        # Test each obstacle individually
        for obs_idx, obstacle in enumerate(self.obstacles):
            print(f"  Testing obstacle {obs_idx + 1}/{len(self.obstacles)}...")
            
            # Temporarily create robot with only this obstacle
            single_obstacle_robot = DPlanarRobot(
                links=self.links,
                obstacles=[obstacle],
                tensor_args=self.tensor_args
            )
            single_obstacle_robot.reset(num_particles=1)
            
            # Test all configurations against this single obstacle
            collision_mask = torch.zeros(N, dtype=torch.bool, device=self.tensor_args['device'])
            
            batch_size = 1000  # Process in batches to save memory
            for i in range(0, N, batch_size):
                end_idx = min(i + batch_size, N)
                batch_configs = config_grid[i:end_idx]
                
                for j, config in enumerate(batch_configs):
                    # Set robot configuration
                    single_obstacle_robot.x = torch.cat([config, torch.zeros(len(config), **self.tensor_args)]).unsqueeze(0)
                    
                    # Check collision with workspace test points
                    arm_length = sum([l.length for l in self.links])
                    test_points = self._create_workspace_test_points(arm_length, resolution=30, adaptive=True)
                    
                    robot_sdf = single_obstacle_robot.sdf_at_points(test_points)
                    env_sdf = single_obstacle_robot.environment_sdf_at_points(test_points)
                    
                    # Check for collision
                    robot_inside = robot_sdf[0] < 0
                    env_inside = env_sdf < 0
                    collision_points = robot_inside & env_inside
                    
                    collision_mask[i + j] = torch.any(collision_points)
            
            obstacle_collision_masks.append(collision_mask)
        
        # Compute combined collision mask
        combined_collision_mask = torch.zeros(N, dtype=torch.bool, device=self.tensor_args['device'])
        for mask in obstacle_collision_masks:
            combined_collision_mask |= mask
        
        # Compute SDF values using full environment
        print("Computing SDF values for all configurations...")
        sdf_values = torch.zeros(N, **self.tensor_args)
        
        batch_size = 1000
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_configs = config_grid[i:end_idx]
            
            for j, config in enumerate(batch_configs):
                # Set robot configuration
                self.x = torch.cat([config, torch.zeros(len(config), **self.tensor_args)]).unsqueeze(0)
                
                # Compute minimum SDF
                arm_length = sum([l.length for l in self.links])
                test_points = self._create_workspace_test_points(arm_length, resolution=25, adaptive=True)
                
                robot_sdf = self.sdf_at_points(test_points)
                env_sdf = self.environment_sdf_at_points(test_points)
                
                # Minimum distance between robot and environment
                distances = robot_sdf[0] + env_sdf
                sdf_values[i + j] = torch.min(distances)
        
        return config_grid, obstacle_collision_masks, combined_collision_mask, sdf_values

    def _create_workspace_test_points(self, arm_length, resolution=20, adaptive=True):
        """
        Create a grid of test points covering the robot workspace.
        
        Args:
            arm_length: Maximum reach of the robot
            resolution: Base resolution for uniform grid
            adaptive: If True, add dense sampling around obstacles
        """
        margin = 0.2
        lower = -arm_length - margin
        upper = arm_length + margin
        
        # Base uniform grid
        xs = torch.linspace(lower, upper, resolution, **self.tensor_args)
        ys = torch.linspace(lower, upper, resolution, **self.tensor_args)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        base_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        if not adaptive or len(self.obstacles) == 0:
            return base_points
        
        # Add dense sampling around obstacles for more robust collision detection
        adaptive_points = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center = obstacle.center
                radius = obstacle.radius.item()
                
                # Create dense grid around obstacle
                dense_res = max(10, resolution // 3)
                margin_factor = 1.5  # Sample beyond obstacle boundary
                local_margin = radius * margin_factor
                
                local_xs = torch.linspace(
                    center[0] - local_margin, center[0] + local_margin, 
                    dense_res, **self.tensor_args
                )
                local_ys = torch.linspace(
                    center[1] - local_margin, center[1] + local_margin, 
                    dense_res, **self.tensor_args
                )
                Local_X, Local_Y = torch.meshgrid(local_xs, local_ys, indexing='ij')
                local_points = torch.stack([Local_X.flatten(), Local_Y.flatten()], dim=1)
                adaptive_points.append(local_points)
                
            elif isinstance(obstacle, RectangleObstacle):
                # Get rectangle corners and create dense sampling around it
                corners = torch.tensor(obstacle.get_corners_numpy(), **self.tensor_args)
                
                # Find bounding box
                min_x, max_x = torch.min(corners[:, 0]), torch.max(corners[:, 0])
                min_y, max_y = torch.min(corners[:, 1]), torch.max(corners[:, 1])
                
                # Add margin
                margin_rect = max(obstacle.width, obstacle.height) * 0.2
                
                dense_res = max(15, resolution // 2)
                local_xs = torch.linspace(min_x - margin_rect, max_x + margin_rect, dense_res, **self.tensor_args)
                local_ys = torch.linspace(min_y - margin_rect, max_y + margin_rect, dense_res, **self.tensor_args)
                Local_X, Local_Y = torch.meshgrid(local_xs, local_ys, indexing='ij')
                local_points = torch.stack([Local_X.flatten(), Local_Y.flatten()], dim=1)
                adaptive_points.append(local_points)
        
        # Combine all points
        if adaptive_points:
            all_points = torch.cat([base_points] + adaptive_points, dim=0)
            # Remove duplicates (approximately)
            unique_points = self._remove_duplicate_points(all_points, tolerance=0.01)
            return unique_points
        else:
            return base_points
    
    def _remove_duplicate_points(self, points, tolerance=0.01):
        """Remove approximately duplicate points from a tensor."""
        if points.shape[0] == 0:
            return points
            
        # Simple approach: keep points that are sufficiently far from all previous points
        unique_points = [points[0:1]]  # Start with first point
        
        for i in range(1, points.shape[0]):
            current_point = points[i:i+1]
            
            # Check distance to all existing unique points
            distances = torch.norm(torch.cat(unique_points, dim=0) - current_point, dim=1)
            
            # If far enough from all existing points, keep it
            if torch.min(distances) > tolerance:
                unique_points.append(current_point)
                
            # Limit total points to avoid excessive computation
            if len(unique_points) > 2000:  # Reasonable upper limit
                break
        
        return torch.cat(unique_points, dim=0)
    
    def enhanced_collision_check(self, config, safety_margin=0.02, use_conservative=True):
        """
        Enhanced collision detection with multiple strategies.
        
        Args:
            config: Joint configuration tensor (num_actuated_joints,)
            safety_margin: Additional safety buffer in meters
            use_conservative: If True, use conservative collision detection
            
        Returns:
            collision_detected: Boolean indicating collision
            min_distance: Minimum distance to obstacles
            collision_info: Dict with detailed collision information
        """
        if len(self.obstacles) == 0:
            return False, float('inf'), {'method': 'no_obstacles'}
        
        # Save current state
        original_state = self.x.clone()
        
        try:
            # Set robot to test configuration (zero velocity)
            test_state = torch.cat([
                config.unsqueeze(0), 
                torch.zeros(1, len(config), **self.tensor_args)
            ], dim=1)
            self.x = test_state
            
            arm_length = sum([link.length for link in self.links])
            
            # Strategy 1: High-resolution workspace sampling
            test_points_dense = self._create_workspace_test_points(
                arm_length, resolution=60, adaptive=True
            )
            
            robot_sdf = self.sdf_at_points(test_points_dense)[0]  # Remove batch dimension
            env_sdf = self.environment_sdf_at_points(test_points_dense)
            
            # Strategy 2: Conservative collision detection with safety margin
            if use_conservative:
                # Consider collision if robot is within safety margin of obstacles
                robot_near_surface = robot_sdf <= safety_margin
                env_collision_zone = env_sdf <= safety_margin
                
                collision_points = robot_near_surface & env_collision_zone
                collision_detected = torch.any(collision_points)
            else:
                # Standard collision detection
                robot_inside = robot_sdf < 0
                env_inside = env_sdf < 0
                collision_points = robot_inside & env_inside
                collision_detected = torch.any(collision_points)
            
            # Strategy 3: Compute minimum distance for validation
            robot_boundary_points = test_points_dense[robot_sdf <= 0.1]  # Points near robot
            if robot_boundary_points.numel() > 0:
                boundary_distances = self.environment_sdf_at_points(robot_boundary_points)
                min_distance = torch.min(boundary_distances).item()
            else:
                min_distance = float('inf')
            
            # Collect collision information
            collision_info = {
                'method': 'enhanced',
                'num_test_points': test_points_dense.shape[0],
                'num_collision_points': torch.sum(collision_points).item(),
                'safety_margin': safety_margin,
                'conservative': use_conservative,
                'min_distance': min_distance
            }
            
            return collision_detected.item(), min_distance, collision_info
            
        finally:
            # Restore original state
            self.x = original_state

    def plot_configuration_space_2d(self, joint_ranges, resolution=50, show_robot_configs=True):
        """
        Plot 2D configuration space for a 2-DOF robot.
        
        Args:
            joint_ranges: List of two tuples [(min1, max1), (min2, max2)]
            resolution: Grid resolution
            show_robot_configs: Whether to show sample robot configurations
            
        Returns:
            fig: matplotlib figure
        """
        if len(self.actuated_indices) != 2:
            raise ValueError("This method is only for 2-DOF robots")
        
        # Compute C-space
        config_grid, collision_mask, sdf_values = self.compute_configuration_space(
            joint_ranges, resolution
        )
        
        # Reshape for plotting
        collision_grid = collision_mask.reshape(resolution, resolution)
        sdf_grid = sdf_values.reshape(resolution, resolution)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Collision map
        ax1.set_title("Configuration Space - Collision Map", fontsize=14)
        ax1.set_xlabel(f"Joint 1 (rad)")
        ax1.set_ylabel(f"Joint 2 (rad)")
        
        extent = [joint_ranges[0][0], joint_ranges[0][1], 
                 joint_ranges[1][0], joint_ranges[1][1]]
        
        # Show collisions in red, free space in green
        collision_colors = collision_grid.cpu().numpy().astype(float)
        im1 = ax1.imshow(collision_colors.T, extent=extent, origin='lower', 
                        cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: SDF values (clearance)
        ax2.set_title("Configuration Space - Clearance Values", fontsize=14)
        ax2.set_xlabel(f"Joint 1 (rad)")
        ax2.set_ylabel(f"Joint 2 (rad)")
        
        # Mask collision regions for SDF plot
        sdf_plot = sdf_grid.cpu().numpy()
        sdf_plot[collision_grid.cpu().numpy()] = -0.1  # Show collisions as negative
        
        im2 = ax2.imshow(sdf_plot.T, extent=extent, origin='lower', 
                        cmap='viridis', alpha=0.8)
        fig.colorbar(im2, ax=ax2, label='Clearance Distance')
        ax2.grid(True, alpha=0.3)
        
        # Add sample robot configurations if requested
        if show_robot_configs and len(self.obstacles) > 0:
            # Show a few example configurations
            sample_indices = [
                (resolution//4, resolution//4),
                (3*resolution//4, resolution//4),
                (resolution//4, 3*resolution//4),
                (3*resolution//4, 3*resolution//4),
                (resolution//2, resolution//2)
            ]
            
            # Create small subplots for robot configurations
            for idx, (i, j) in enumerate(sample_indices):
                if i < resolution and j < resolution:
                    # Get configuration
                    config_idx = i * resolution + j
                    config = config_grid[config_idx]
                    is_collision = collision_mask[config_idx]
                    
                    # Create inset axis
                    inset_size = 0.15
                    inset_x = 0.02 + (idx % 3) * 0.32
                    inset_y = 0.02 if idx < 3 else 0.52
                    
                    ax_inset = fig.add_axes([inset_x, inset_y, inset_size, inset_size])
                    ax_inset.set_aspect('equal')
                    ax_inset.set_xlim(-sum([l.length for l in self.links])*1.1, 
                                     sum([l.length for l in self.links])*1.1)
                    ax_inset.set_ylim(-sum([l.length for l in self.links])*1.1, 
                                     sum([l.length for l in self.links])*1.1)
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    
                    # Set robot configuration and plot
                    temp_x = self.x.clone()
                    self.x = torch.cat([config, torch.zeros(len(config), **self.tensor_args)]).unsqueeze(0)
                    
                    # Plot obstacles
                    for obstacle in self.obstacles:
                        if isinstance(obstacle, CircleObstacle):
                            center = obstacle.center.cpu().numpy()
                            radius = obstacle.radius.item()
                            circle = Circle(center, radius, fill=True, color='red', alpha=0.5)
                            ax_inset.add_patch(circle)
                        elif isinstance(obstacle, RectangleObstacle):
                            center = obstacle.center.cpu().numpy()
                            width = obstacle.width.item()
                            height = obstacle.height.item()
                            angle_deg = np.degrees(obstacle.angle.item())
                            
                            # Use precomputed rotated bottom-left corner
                            bottom_left = obstacle.get_matplotlib_bottom_left()
                            
                            rect = Rectangle(bottom_left, width, height, angle=angle_deg,
                                           fill=True, color='red', alpha=0.5)
                            ax_inset.add_patch(rect)
                    
                    # Plot robot
                    full_angles = self._build_full_angle_vector(config.unsqueeze(0))
                    origins, endpoints, circle_positions = self.forward_kinematics(full_angles)
                    
                    robot_color = 'red' if is_collision else 'blue'
                    for link_idx, link in enumerate(self.links):
                        ox, oy = origins[0, link_idx, :].cpu().numpy()
                        ex, ey = endpoints[0, link_idx, :].cpu().numpy()
                        ax_inset.plot([ox, ex], [oy, ey], linewidth=2, color=robot_color)
                        ax_inset.scatter([ex], [ey], s=20, color=robot_color)
                    
                    # Mark configuration on C-space plots
                    q1, q2 = config.cpu().numpy()
                    ax1.plot(q1, q2, 'ko' if is_collision else 'wo', markersize=8, 
                            markeredgecolor='black', markeredgewidth=1)
                    ax2.plot(q1, q2, 'ko' if is_collision else 'wo', markersize=8,
                            markeredgecolor='black', markeredgewidth=1)
                    
                    # Restore state
                    self.x = temp_x
        
        plt.tight_layout()
        return fig

    def plot_configuration_space_per_obstacle(self, joint_ranges, resolution=50):
        """
        Plot configuration space showing individual obstacle contributions.
        Each obstacle gets its own color in both workspace and C-space.
        
        Args:
            joint_ranges: List of (min, max) tuples for each joint
            resolution: Resolution for each joint dimension
            
        Returns:
            fig: matplotlib figure with subplots
        """
        if len(joint_ranges) != 2:
            raise ValueError("Currently only supports 2-DOF robots")
        
        # Compute per-obstacle C-space
        config_grid, obstacle_masks, combined_mask, sdf_values = self.compute_configuration_space_per_obstacle(
            joint_ranges, resolution
        )
        
        # Reshape for plotting
        joint1_range, joint2_range = joint_ranges
        collision_grids = []
        for mask in obstacle_masks:
            collision_grid = mask.view(resolution, resolution).cpu().numpy()
            collision_grids.append(collision_grid)
        
        combined_grid = combined_mask.view(resolution, resolution).cpu().numpy()
        
        # Create figure with subplots
        num_obstacles = len(self.obstacles)
        fig, axes = plt.subplots(2, num_obstacles + 1, figsize=(5 * (num_obstacles + 1), 10))
        
        if num_obstacles == 1:
            axes = axes.reshape(2, 2)  # Ensure 2D array
        
        # Plot workspace with obstacles (top row)
        arm_length = sum([l.length for l in self.links])
        
        for obs_idx in range(num_obstacles):
            ax = axes[0, obs_idx]
            self._plot_workspace_with_single_obstacle(ax, obs_idx, arm_length)
            ax.set_title(f'Obstacle {obs_idx + 1}: {self.obstacles[obs_idx].color}')
        
        # Combined workspace
        ax = axes[0, num_obstacles]
        self._plot_workspace_with_all_obstacles(ax, arm_length)
        ax.set_title('All Obstacles')
        
        # Plot C-space contributions (bottom row)
        extent = [joint1_range[0], joint1_range[1], joint2_range[0], joint2_range[1]]
        
        for obs_idx in range(num_obstacles):
            ax = axes[1, obs_idx]
            
            # Create colored collision map for this obstacle
            collision_map = np.zeros((resolution, resolution, 4))  # RGBA
            obstacle_color = plt.cm.colors.to_rgba(self.obstacles[obs_idx].color, alpha=0.8)
            
            mask = collision_grids[obs_idx]
            collision_map[mask] = obstacle_color
            
            # Show collision regions
            ax.imshow(collision_map, extent=extent, origin='lower', aspect='auto')
            ax.set_xlabel('Joint 1 (rad)')
            ax.set_ylabel('Joint 2 (rad)')
            ax.set_title(f'C-space Obstacle {obs_idx + 1}')
            ax.grid(True, alpha=0.3)
        
        # Combined C-space
        ax = axes[1, num_obstacles]
        
        # Create multi-colored collision map
        combined_collision_map = np.zeros((resolution, resolution, 4))
        
        for obs_idx in range(num_obstacles):
            obstacle_color = plt.cm.colors.to_rgba(self.obstacles[obs_idx].color, alpha=0.6)
            mask = collision_grids[obs_idx]
            
            # Blend colors where obstacles overlap
            combined_collision_map[mask] += np.array(obstacle_color)
        
        # Normalize to prevent over-saturation
        combined_collision_map = np.clip(combined_collision_map, 0, 1)
        
        ax.imshow(combined_collision_map, extent=extent, origin='lower', aspect='auto')
        ax.set_xlabel('Joint 1 (rad)')
        ax.set_ylabel('Joint 2 (rad)')
        ax.set_title('Combined C-space')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        total_configs = len(combined_mask)
        combined_collisions = torch.sum(combined_mask).item()
        
        fig.suptitle(f'Per-Obstacle Configuration Space Analysis\\n'
                    f'Resolution: {resolution}×{resolution}, '
                    f'Collision Rate: {combined_collisions/total_configs*100:.1f}%', 
                    fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def _plot_workspace_with_single_obstacle(self, ax, obs_idx, arm_length):
        """Helper method to plot workspace with a single obstacle highlighted."""
        limit = arm_length * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Plot all obstacles with reduced alpha, except the highlighted one
        for i, obstacle in enumerate(self.obstacles):
            alpha = 0.8 if i == obs_idx else 0.2
            color = obstacle.color if i == obs_idx else 'gray'
            
            if isinstance(obstacle, CircleObstacle):
                center = obstacle.center.cpu().numpy()
                radius = obstacle.radius.item()
                circle = plt.Circle(center, radius, fill=True, color=color, alpha=alpha, 
                                  edgecolor='black', linewidth=1)
                ax.add_patch(circle)
            elif isinstance(obstacle, RectangleObstacle):
                center = obstacle.center.cpu().numpy()
                width = obstacle.width.item()
                height = obstacle.height.item()
                angle_deg = np.degrees(obstacle.angle.item())
                
                # Use precomputed rotated bottom-left corner
                bottom_left = obstacle.get_matplotlib_bottom_left()
                
                rect = plt.Rectangle(bottom_left, width, height, angle=angle_deg,
                                   fill=True, color=color, alpha=alpha, 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(rect)
    
    def _plot_workspace_with_all_obstacles(self, ax, arm_length):
        """Helper method to plot workspace with all obstacles."""
        limit = arm_length * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Plot all obstacles with their colors
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center = obstacle.center.cpu().numpy()
                radius = obstacle.radius.item()
                circle = plt.Circle(center, radius, fill=True, color=obstacle.color, alpha=0.6, 
                                  edgecolor='black', linewidth=1)
                ax.add_patch(circle)
            elif isinstance(obstacle, RectangleObstacle):
                center = obstacle.center.cpu().numpy()
                width = obstacle.width.item()
                height = obstacle.height.item()
                angle_deg = np.degrees(obstacle.angle.item())
                
                # Use precomputed rotated bottom-left corner
                bottom_left = obstacle.get_matplotlib_bottom_left()
                
                rect = plt.Rectangle(bottom_left, width, height, angle=angle_deg,
                                   fill=True, color=obstacle.color, alpha=0.6, 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(rect)