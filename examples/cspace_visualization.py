#!/usr/bin/env python3
"""
Configuration Space Visualization Script

This script demonstra    # Compute per-obstacle C-space contributions
    resolution = 100  # Higher resolution for more accurate visualization
    print(f"Using resolution: {resolution}x{resolution} = {resolution*resolution} configurations"):
1. Computing the configuration space for a planar robot with obstacles
2. Visualizing each obstacle's contribution to C-space with different colors
3. Showing combined C-space and collision-free regions
4. Interactive visualization of robot configurations in both C-space and workspace
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from loud_robotics.environments.planar_robot import (
    DPlanarRobot, 
    Link, 
    CircleObstacle, 
    RectangleObstacle
)

def main():
    # Set up device and random seed
    tensor_args = {'dtype': torch.float32, 'device': 'cpu'}
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Creating planar robot for C-space analysis...")
    
    # Create robot links with proper angle limits - 3 DOF robot as specified
    links = [
        Link(
            length=0.8, 
            circle_offsets=[0.15, 0.4, 0.65], 
            circle_radii=[0.15, 0.15, 0.15],
            tensor_args=tensor_args,
            fixed=True
        ),
        Link(
            length=0.8, 
            circle_offsets=[0.15, 0.4, 0.65], 
            circle_radii=[0.15, 0.15, 0.15],
            tensor_args=tensor_args,
            angle_limits=(-np.pi, np.pi)  # Joint 1 limits - full rotation
        ),
        Link(
            length=0.8, 
            circle_offsets=[0.15, 0.4, 0.65], 
            circle_radii=[0.15, 0.15, 0.15],
            tensor_args=tensor_args,
            angle_limits=(-np.pi, np.pi)  # Joint 2 limits - full rotation  
        )
    ]
    
    # Create obstacles as specified
    obstacles = [
        # Rectangular obstacles
        RectangleObstacle(center=(1.0, 1.0), width=1.3, height=0.1, 
                         angle=0.0, tensor_args=tensor_args, color='red'),
        RectangleObstacle(center=(1.0, 0.475), width=0.1, height=0.95, 
                         angle=0.0, tensor_args=tensor_args, color='blue'),
        RectangleObstacle(center=(0, -1.0), width=10, height=2.0, 
                         tensor_args=tensor_args, color='green'),
    ]
    
    # Initialize robot with obstacles
    robot = DPlanarRobot(
        links=links,
        obstacles=obstacles,
        starting_angle_config=torch.tensor([np.pi/2.0, -0.5, 0.8], **tensor_args),
        tensor_args=tensor_args,
        seed=42
    )
    
    print(f"Robot created with {len(robot.obstacles)} obstacles")
    
    # Reset robot to initial state
    robot.reset(num_particles=1)
    
    # Extract joint ranges from link angle limits
    joint_ranges = []
    for i, link in enumerate(robot.links):
        if not link.fixed and link.angle_limits is not None:
            joint_ranges.append(link.angle_limits)
    
    # If no angle limits defined, use default ranges
    if len(joint_ranges) == 0:
        joint_ranges = [
            (-np.pi, np.pi),      # Joint 1 range - full rotation
            (-np.pi, np.pi),      # Joint 2 range - full rotation  
        ]
    
    print(f"\nComputing C-space for {len(robot.actuated_indices)} actuated joints...")
    print(f"Joint ranges from link limits: {joint_ranges}")
    
    # Compute per-obstacle C-space contributions with fine sampling
    resolution = 500  # Higher resolution for accuracy - not concerned about efficiency
    print(f"Using resolution: {resolution}x{resolution} = {resolution*resolution} configurations")
    
    # Use our enhanced sampling approach with proper joint limits
    print("Computing C-space with enhanced joint sampling...")
    config_grid, obstacle_masks, combined_mask, sdf_values = compute_enhanced_cspace_per_obstacle(
        robot, joint_ranges, resolution, tensor_args
    )
    
    print(f"\nResults:")
    print(f"  Total configurations: {len(combined_mask)}")
    
    for obs_idx, obstacle in enumerate(robot.obstacles):
        collision_count = torch.sum(obstacle_masks[obs_idx]).item()
        collision_rate = collision_count / len(obstacle_masks[obs_idx]) * 100
        print(f"  Obstacle {obs_idx + 1} ({obstacle.color}): {collision_count} collisions ({collision_rate:.1f}%)")
    
    total_collisions = torch.sum(combined_mask).item()
    overall_rate = total_collisions / len(combined_mask) * 100
    print(f"  Combined: {total_collisions} collisions ({overall_rate:.1f}%)")
    
    # Create enhanced visualization
    create_enhanced_cspace_visualization(
        robot, config_grid, obstacle_masks, combined_mask, sdf_values,
        joint_ranges, resolution, tensor_args
    )
    
    # Also create the built-in visualization for comparison
    print("\nGenerating built-in visualization...")
    fig_builtin = robot.plot_configuration_space_per_obstacle(joint_ranges, resolution)
    fig_builtin.savefig("cspace_builtin_comparison.pdf", bbox_inches='tight', dpi=300)
    print("✓ Built-in visualization saved as cspace_builtin_comparison.pdf")
    
    print("\nC-space visualization completed!")
    print("Generated files:")
    print("  - cspace_enhanced_analysis.pdf")
    print("  - cspace_per_obstacle.pdf") 
    print("  - cspace_combined.pdf")
    print("  - cspace_builtin_comparison.pdf")


def compute_enhanced_cspace_per_obstacle(robot, joint_ranges, resolution, tensor_args):
    """
    Compute configuration space with enhanced sampling that properly samples 
    each joint from its lower to upper limits.
    """
    print(f"Setting up enhanced joint sampling...")
    
    # Create proper joint value arrays for each actuated joint
    joint_values = []
    for i, (joint_min, joint_max) in enumerate(joint_ranges):
        joint_vals = torch.linspace(joint_min, joint_max, resolution, **tensor_args)
        joint_values.append(joint_vals)
        print(f"  Joint {i+1}: {resolution} samples from {joint_min:.3f} to {joint_max:.3f}")
    
    # Create configuration grid - handle 2D and 3D cases
    if len(joint_ranges) == 2:
        # 2D case - create meshgrid
        j1_grid, j2_grid = torch.meshgrid(joint_values[0], joint_values[1], indexing='ij')
        config_grid = torch.stack([j1_grid.flatten(), j2_grid.flatten()], dim=1)
        print(f"Created 2D configuration grid: {config_grid.shape}")
        
    elif len(joint_ranges) == 3:
        # 3D case - for visualization we'll project to first 2 joints
        j1_grid, j2_grid, j3_grid = torch.meshgrid(
            joint_values[0], joint_values[1], joint_values[2], indexing='ij'
        )
        config_grid = torch.stack([j1_grid.flatten(), j2_grid.flatten(), j3_grid.flatten()], dim=1)
        print(f"Created 3D configuration grid: {config_grid.shape}")
        
    else:
        raise ValueError(f"Currently supports 2-3 DOF robots, got {len(joint_ranges)} joints")
    
    total_configs = config_grid.shape[0]
    print(f"Total configurations to evaluate: {total_configs:,}")
    
    # Compute combined C-space first
    print("\nComputing combined C-space with all obstacles...")
    combined_mask, sdf_values = evaluate_configurations_for_collision(
        robot, config_grid, tensor_args, "All obstacles"
    )
    
    # Compute individual obstacle contributions
    print("\nComputing individual obstacle contributions...")
    obstacle_masks = []
    
    for obs_idx, obstacle in enumerate(robot.obstacles):
        print(f"\n  Processing obstacle {obs_idx + 1}/{len(robot.obstacles)}: {obstacle.color}")
        
        # Create temporary robot with only this obstacle
        temp_robot = DPlanarRobot(
            links=robot.links,
            obstacles=[obstacle],
            tensor_args=tensor_args,
            seed=42
        )
        temp_robot.reset(num_particles=1)
        
        # Evaluate configurations for this obstacle only
        obs_mask, _ = evaluate_configurations_for_collision(
            temp_robot, config_grid, tensor_args, f"{obstacle.color} obstacle"
        )
        obstacle_masks.append(obs_mask)
        
        # Report statistics
        collision_count = torch.sum(obs_mask).item()
        collision_rate = collision_count / total_configs * 100
        print(f"    → {collision_count:,} collisions ({collision_rate:.1f}%)")
    
    print(f"\n✓ Enhanced C-space computation completed!")
    return config_grid, obstacle_masks, combined_mask, sdf_values


def evaluate_configurations_for_collision(robot, config_grid, tensor_args, description):
    """
    Evaluate a grid of configurations for collisions and compute clearance.
    """
    total_configs = config_grid.shape[0]
    batch_size = 500  # Smaller batches for memory management
    
    collision_mask = torch.zeros(total_configs, dtype=torch.bool, device=tensor_args['device'])
    sdf_values = torch.zeros(total_configs, device=tensor_args['device'], dtype=tensor_args['dtype'])
    
    print(f"    Evaluating {total_configs:,} configurations for {description}...")
    
    # Store original robot state
    original_x = robot.x.clone()
    original_particles = robot.num_particles
    
    num_batches = (total_configs + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_configs)
        batch_configs = config_grid[start_idx:end_idx]
        batch_size_actual = batch_configs.shape[0]
        
        # Progress reporting
        progress = (end_idx / total_configs) * 100
        if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
            print(f"      Progress: {progress:.1f}% ({end_idx:,}/{total_configs:,})")
        
        # Prepare robot configurations for this batch
        full_configs = torch.zeros(batch_size_actual, len(robot.links), device=tensor_args['device'], dtype=tensor_args['dtype'])
        
        # Set actuated joint values
        for i, joint_idx in enumerate(robot.actuated_indices):
            if i < batch_configs.shape[1]:  # Ensure we don't exceed available configs
                full_configs[:, joint_idx] = batch_configs[:, i]
        
        # Create robot state (position and velocity)
        batch_states = torch.cat([
            full_configs,
            torch.zeros(batch_size_actual, len(robot.links), device=tensor_args['device'], dtype=tensor_args['dtype'])  # Zero velocities
        ], dim=1)
        
        # Set robot state
        robot.x = batch_states
        robot.num_particles = batch_size_actual
        
        # Evaluate collisions and clearance
        if len(robot.obstacles) > 0:
            # Check for self-collision first (if applicable)
            has_self_collision = torch.zeros(batch_size_actual, dtype=torch.bool, device=tensor_args['device'])
            
            # Check environment collisions
            arm_length = sum([l.length for l in robot.links])
            test_points = robot._create_workspace_test_points(arm_length, resolution=50, adaptive=True)
            
            # Compute robot and environment SDFs
            robot_sdf = robot.sdf_at_points(test_points)  # (batch_size, num_test_points)
            env_sdf = robot.environment_sdf_at_points(test_points)  # (num_test_points,)
            
            # Collision detection: robot SDF < 0 AND environment SDF < 0
            robot_inside = robot_sdf < 0  # Points inside robot
            env_inside = env_sdf < 0      # Points inside obstacles
            
            # Collision if any test point is inside both robot and environment
            collision_points = robot_inside & env_inside.unsqueeze(0)
            batch_collisions = torch.any(collision_points, dim=1)
            
            # Compute clearance (minimum distance between robot and obstacles)
            min_clearance = torch.full((batch_size_actual,), float('inf'), device=tensor_args['device'], dtype=tensor_args['dtype'])
            
            for i in range(batch_size_actual):
                robot_boundary = robot_sdf[i] <= 0.02  # Points near robot surface
                if torch.any(robot_boundary):
                    robot_points = test_points[robot_boundary]
                    clearances = robot.environment_sdf_at_points(robot_points)
                    min_clearance[i] = torch.min(clearances)
            
            collision_mask[start_idx:end_idx] = batch_collisions | has_self_collision
            sdf_values[start_idx:end_idx] = min_clearance
            
        else:
            # No obstacles, no collisions (but might have self-collisions)
            sdf_values[start_idx:end_idx] = float('inf')
    
    # Restore original robot state
    robot.x = original_x
    robot.num_particles = original_particles
    
    collision_count = torch.sum(collision_mask).item()
    collision_rate = collision_count / total_configs * 100
    print(f"    ✓ {description}: {collision_count:,} collisions ({collision_rate:.1f}%)")
    
    return collision_mask, sdf_values


def create_enhanced_cspace_visualization(robot, config_grid, obstacle_masks, combined_mask, sdf_values,
                               joint_ranges, resolution, tensor_args):
    """Create enhanced C-space visualization that integrates with existing methods."""
    
    # Use matplotlib for enhanced styling
    plt.style.use('default')
    
    # Define colors for each obstacle
    obstacle_colors = [obs.color for obs in robot.obstacles]
    
    # Reshape grids for plotting
    joint1_range, joint2_range = joint_ranges
    
    # Create main comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('Enhanced Configuration Space Analysis with Obstacle Contributions', fontsize=20, y=0.95)
    
    # Calculate grid layout
    num_obstacles = len(robot.obstacles)
    rows = 4
    cols = max(4, num_obstacles)
    
    # Row 1: Individual obstacle workspace visualizations
    for obs_idx in range(num_obstacles):
        ax = plt.subplot(rows, cols, obs_idx + 1)
        plot_single_obstacle_workspace(robot, obs_idx, ax)
        ax.set_title(f'Workspace: Obstacle {obs_idx + 1} ({robot.obstacles[obs_idx].color})', fontsize=12)
    
    # Row 2: Individual obstacle C-space contributions  
    extent = [joint1_range[0], joint1_range[1], joint2_range[0], joint2_range[1]]
    for obs_idx in range(num_obstacles):
        ax = plt.subplot(rows, cols, cols + obs_idx + 1)
        
        # Create high-contrast visualization
        collision_grid = obstacle_masks[obs_idx].view(resolution, resolution).cpu().numpy()
        
        # Use obstacle-specific colormap
        obstacle_color = robot.obstacles[obs_idx].color
        cmap = mcolors.ListedColormap(['white', obstacle_color])
        
        im = ax.imshow(collision_grid.T, extent=extent, origin='lower', 
                      cmap=cmap, aspect='auto', alpha=0.9)
        ax.set_xlabel('Joint 1 (rad)', fontsize=10)
        ax.set_ylabel('Joint 2 (rad)', fontsize=10)
        ax.set_title(f'C-space: Obstacle {obs_idx + 1}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add detailed statistics
        collision_count = torch.sum(obstacle_masks[obs_idx]).item()
        total_configs = len(obstacle_masks[obs_idx])
        collision_rate = collision_count / total_configs * 100
        free_rate = 100 - collision_rate
        
        # Statistics box
        stats_text = f'Collision: {collision_rate:.1f}%\\nFree: {free_rate:.1f}%\\nCount: {collision_count}/{total_configs}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                verticalalignment='top', fontsize=9)
    
    # Row 3: Combined analysis
    # Combined workspace 
    ax_workspace = plt.subplot(rows, cols, 2*cols + 1)
    plot_combined_workspace(robot, ax_workspace, tensor_args)
    ax_workspace.set_title('Combined Workspace', fontsize=12)
    
    # Combined C-space with overlaps
    ax_combined = plt.subplot(rows, cols, 2*cols + 2)
    plot_combined_cspace_with_overlaps(robot, obstacle_masks, combined_mask, extent, resolution, ax_combined)
    ax_combined.set_title('Combined C-space with Overlaps', fontsize=12)
    
    # Clearance visualization
    ax_clearance = plt.subplot(rows, cols, 2*cols + 3)
    plot_clearance_analysis(sdf_values, combined_mask, extent, resolution, ax_clearance)
    
    # Configuration sampling analysis
    ax_sampling = plt.subplot(rows, cols, 2*cols + 4)
    plot_configuration_sampling_analysis(robot, config_grid, combined_mask, joint_ranges, resolution, ax_sampling, tensor_args)
    
    # Row 4: Advanced analysis
    # Joint correlation analysis
    ax_correlation = plt.subplot(rows, cols, 3*cols + 1)
    plot_joint_correlation_analysis(config_grid, combined_mask, joint_ranges, resolution, ax_correlation)
    
    # Collision density heatmap
    ax_density = plt.subplot(rows, cols, 3*cols + 2)
    plot_collision_density_heatmap(obstacle_masks, robot.obstacles, extent, resolution, ax_density)
    
    # Path planning feasibility
    ax_paths = plt.subplot(rows, cols, 3*cols + 3)
    plot_path_feasibility_analysis(config_grid, combined_mask, joint_ranges, resolution, ax_paths)
    
    # Summary statistics
    ax_summary = plt.subplot(rows, cols, 3*cols + 4)
    plot_summary_statistics(robot, obstacle_masks, combined_mask, ax_summary)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    plt.savefig("cspace_enhanced_analysis.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also create the detailed plots
    create_detailed_cspace_plots(robot, obstacle_masks, combined_mask, sdf_values,
                                joint_ranges, resolution)


def plot_single_obstacle_workspace(robot, obs_idx, ax):
    """Plot workspace with a single obstacle highlighted."""
    arm_length = sum([l.length for l in robot.links])
    ax.set_xlim(-arm_length*1.1, arm_length*1.1)
    ax.set_ylim(-arm_length*0.5, arm_length*1.1)
    ax.set_aspect('equal')
    
    # Plot only the specific obstacle
    obstacle = robot.obstacles[obs_idx]
    if isinstance(obstacle, CircleObstacle):
        center = obstacle.center.cpu().numpy()
        radius = obstacle.radius.item()
        circle = Circle(center, radius, fill=True, color=obstacle.color, 
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
    elif isinstance(obstacle, RectangleObstacle):
        center = obstacle.center.cpu().numpy()
        width = obstacle.width.item()
        height = obstacle.height.item()
        angle_deg = np.degrees(obstacle.angle.item())
        
        bottom_left = obstacle.get_matplotlib_bottom_left()
        rect = Rectangle(bottom_left, width, height, angle=angle_deg, 
                       fill=True, color=obstacle.color, alpha=0.7, 
                       edgecolor='black', linewidth=2)
        ax.add_patch(rect)
    
    # Plot robot in current configuration with transparency
    plot_robot_in_workspace(robot, ax, tensor_args={'dtype': torch.float32, 'device': 'cpu'}, alpha=0.5)
    ax.grid(True, alpha=0.3)


def plot_combined_workspace(robot, ax, tensor_args):
    """Plot workspace with all obstacles."""
    arm_length = sum([l.length for l in robot.links])
    ax.set_xlim(-arm_length*1.1, arm_length*1.1)
    ax.set_ylim(-arm_length*0.5, arm_length*1.1)
    ax.set_aspect('equal')
    
    # Draw all obstacles
    for obstacle in robot.obstacles:
        if isinstance(obstacle, CircleObstacle):
            center = obstacle.center.cpu().numpy()
            radius = obstacle.radius.item()
            circle = Circle(center, radius, fill=True, color=obstacle.color, 
                          alpha=0.6, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
        elif isinstance(obstacle, RectangleObstacle):
            center = obstacle.center.cpu().numpy()
            width = obstacle.width.item()
            height = obstacle.height.item()
            angle_deg = np.degrees(obstacle.angle.item())
            
            bottom_left = obstacle.get_matplotlib_bottom_left()
            rect = Rectangle(bottom_left, width, height, angle=angle_deg, 
                           fill=True, color=obstacle.color, alpha=0.6, 
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
    
    # Plot robot
    plot_robot_in_workspace(robot, ax, tensor_args)
    ax.grid(True, alpha=0.3)


def plot_combined_cspace_with_overlaps(robot, obstacle_masks, combined_mask, extent, resolution, ax):
    """Plot combined C-space showing obstacle overlaps."""
    # Create multi-colored collision map showing overlaps
    combined_collision_map = np.ones((resolution, resolution, 4))  # Start with white
    
    # Count overlaps
    overlap_count = np.zeros((resolution, resolution))
    for mask in obstacle_masks:
        collision_grid = mask.view(resolution, resolution).cpu().numpy()
        overlap_count += collision_grid.astype(int)
    
    # Color based on number of overlapping obstacles
    for i in range(resolution):
        for j in range(resolution):
            count = overlap_count[i, j]
            if count == 1:
                # Single obstacle - use obstacle color
                for obs_idx, mask in enumerate(obstacle_masks):
                    if mask.view(resolution, resolution)[i, j]:
                        combined_collision_map[i, j] = mcolors.to_rgba(robot.obstacles[obs_idx].color, alpha=0.7)
                        break
            elif count > 1:
                # Multiple obstacles - use blend or special color
                combined_collision_map[i, j] = mcolors.to_rgba('purple', alpha=min(0.9, count * 0.3))
    
    ax.imshow(combined_collision_map, extent=extent, origin='lower', aspect='auto')
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.grid(True, alpha=0.3)
    
    # Add overlap statistics
    total_configs = len(combined_mask)
    overlap_configs = np.sum(overlap_count > 1)
    overlap_rate = overlap_configs / total_configs * 100
    ax.text(0.02, 0.98, f'Overlaps: {overlap_rate:.1f}%', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            verticalalignment='top')


def plot_clearance_analysis(sdf_values, combined_mask, extent, resolution, ax):
    """Plot clearance analysis with enhanced detail."""
    sdf_plot = sdf_values.view(resolution, resolution).cpu().numpy()
    sdf_plot[combined_mask.view(resolution, resolution).cpu().numpy()] = np.nan
    
    im = ax.imshow(sdf_plot.T, extent=extent, origin='lower', 
                  cmap='viridis', aspect='auto')
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.set_title('Clearance Map', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Clearance Distance')
    
    # Add clearance statistics
    valid_sdf = sdf_values[~combined_mask]
    if len(valid_sdf) > 0:
        mean_clearance = torch.mean(valid_sdf).item()
        min_clearance = torch.min(valid_sdf).item()
        stats_text = f'Mean: {mean_clearance:.3f}\\nMin: {min_clearance:.3f}'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                verticalalignment='bottom')


def plot_configuration_sampling_analysis(robot, config_grid, combined_mask, joint_ranges, resolution, ax, tensor_args):
    """Plot analysis of configuration space sampling quality."""
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.set_title('Sampling Quality Analysis', fontsize=12)
    
    # Show sampling grid
    joint1_vals = torch.linspace(joint_ranges[0][0], joint_ranges[0][1], resolution)
    joint2_vals = torch.linspace(joint_ranges[1][0], joint_ranges[1][1], resolution)
    
    # Create sampling density visualization
    j1_grid, j2_grid = torch.meshgrid(joint1_vals, joint2_vals, indexing='ij')
    
    # Color based on collision status
    collision_grid = combined_mask.view(resolution, resolution).cpu().numpy()
    
    colors = np.where(collision_grid, 'red', 'green')
    alphas = np.where(collision_grid, 0.7, 0.3)
    
    # Plot sample points
    for i in range(0, resolution, max(1, resolution//20)):  # Subsample for visibility
        for j in range(0, resolution, max(1, resolution//20)):
            color = 'red' if collision_grid[i, j] else 'green'
            alpha = 0.7 if collision_grid[i, j] else 0.3
            ax.scatter(j1_grid[i, j], j2_grid[i, j], c=color, alpha=alpha, s=10)
    
    ax.grid(True, alpha=0.3)
    
    # Add sampling statistics
    total_samples = resolution * resolution
    collision_samples = torch.sum(combined_mask).item()
    coverage_text = f'Samples: {total_samples}\\nCollisions: {collision_samples}\\nFree: {total_samples - collision_samples}'
    ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            verticalalignment='top')


def plot_joint_correlation_analysis(config_grid, combined_mask, joint_ranges, resolution, ax):
    """Analyze correlation between joint configurations and collisions."""
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.set_title('Joint Correlation Analysis', fontsize=12)
    
    # Compute collision probability for each joint value
    joint1_vals = torch.linspace(joint_ranges[0][0], joint_ranges[0][1], resolution)
    joint2_vals = torch.linspace(joint_ranges[1][0], joint_ranges[1][1], resolution)
    
    collision_grid = combined_mask.view(resolution, resolution)
    
    # Joint 1 collision probability
    j1_collision_prob = torch.mean(collision_grid.float(), dim=1)  # Average over joint 2
    j2_collision_prob = torch.mean(collision_grid.float(), dim=0)  # Average over joint 1
    
    # Create marginal plots
    ax_top = ax.twinx()
    ax_right = ax.twiny()
    
    ax_top.plot(joint1_vals.cpu().numpy(), j1_collision_prob.cpu().numpy(), 'r-', linewidth=2, alpha=0.7)
    ax_right.plot(j2_collision_prob.cpu().numpy(), joint2_vals.cpu().numpy(), 'b-', linewidth=2, alpha=0.7)
    
    ax_top.set_ylabel('Joint 1 Collision Prob', color='red')
    ax_right.set_xlabel('Joint 2 Collision Prob', color='blue')
    
    # Main heatmap
    im = ax.imshow(collision_grid.T.cpu().numpy(), extent=[joint_ranges[0][0], joint_ranges[0][1], 
                                                          joint_ranges[1][0], joint_ranges[1][1]], 
                  origin='lower', cmap='RdYlGn_r', alpha=0.5, aspect='auto')
    ax.grid(True, alpha=0.3)


def plot_collision_density_heatmap(obstacle_masks, obstacles, extent, resolution, ax):
    """Plot heatmap showing collision density from multiple obstacles."""
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.set_title('Collision Density Heatmap', fontsize=12)
    
    # Sum all obstacle contributions
    density_map = np.zeros((resolution, resolution))
    for mask in obstacle_masks:
        collision_grid = mask.view(resolution, resolution).cpu().numpy()
        density_map += collision_grid.astype(float)
    
    im = ax.imshow(density_map.T, extent=extent, origin='lower', 
                  cmap='hot', aspect='auto')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Obstacle Count')


def plot_path_feasibility_analysis(config_grid, combined_mask, joint_ranges, resolution, ax):
    """Analyze path planning feasibility in C-space."""
    ax.set_xlabel('Joint 1 (rad)')
    ax.set_ylabel('Joint 2 (rad)')
    ax.set_title('Path Feasibility Analysis', fontsize=12)
    
    # Find connected components of free space
    free_space = (~combined_mask).view(resolution, resolution).cpu().numpy()
    
    # Simple connectivity analysis (could be enhanced with proper graph algorithms)
    connectivity_map = np.zeros_like(free_space, dtype=float)
    
    # For each free cell, count nearby free cells
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            if free_space[i, j]:
                # Count free neighbors in 3x3 window
                neighborhood = free_space[i-1:i+2, j-1:j+2]
                connectivity_map[i, j] = np.sum(neighborhood) / 9.0
    
    extent = [joint_ranges[0][0], joint_ranges[0][1], joint_ranges[1][0], joint_ranges[1][1]]
    im = ax.imshow(connectivity_map.T, extent=extent, origin='lower', 
                  cmap='Blues', aspect='auto')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Local Connectivity')


def plot_summary_statistics(robot, obstacle_masks, combined_mask, ax):
    """Plot comprehensive summary statistics."""
    ax.axis('off')
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    total_configs = len(combined_mask)
    total_collisions = torch.sum(combined_mask).item()
    free_configs = total_configs - total_collisions
    
    # Create text summary
    summary_text = f"""
CONFIGURATION SPACE ANALYSIS SUMMARY

Robot Configuration:
• Links: {len(robot.links)} ({len(robot.actuated_indices)} actuated)
• Obstacles: {len(robot.obstacles)}

Sampling Statistics:
• Total configurations: {total_configs:,}
• Collision configurations: {total_collisions:,} ({total_collisions/total_configs*100:.1f}%)
• Free configurations: {free_configs:,} ({free_configs/total_configs*100:.1f}%)

Per-Obstacle Analysis:"""
    
    for obs_idx, obstacle in enumerate(robot.obstacles):
        collision_count = torch.sum(obstacle_masks[obs_idx]).item()
        collision_rate = collision_count / total_configs * 100
        summary_text += f"\n• Obstacle {obs_idx + 1} ({obstacle.color}): {collision_count:,} ({collision_rate:.1f}%)"
    
    # Add recommendations
    if total_collisions / total_configs > 0.8:
        recommendation = "⚠️  High collision rate - consider obstacle reconfiguration"
    elif total_collisions / total_configs > 0.5:
        recommendation = "⚠️  Moderate collision rate - path planning may be challenging"
    else:
        recommendation = "✅ Good collision rate - suitable for motion planning"
    
    summary_text += f"\n\nPath Planning Assessment:\n{recommendation}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

def plot_robot_in_workspace(robot, ax, tensor_args, alpha=1.0):
    """Plot the robot in its current configuration."""
    angles = robot.x[:, :len(robot.actuated_indices)]
    full_angles = robot._build_full_angle_vector(angles)
    origins, endpoints, circle_positions = robot.forward_kinematics(full_angles)
    
    # Plot robot links
    for i, link in enumerate(robot.links):
        ox, oy = origins[0, i, :].cpu().numpy()
        ex, ey = endpoints[0, i, :].cpu().numpy()
        
        # Use different color for each link
        link_color = plt.cm.tab10(i)
        ax.plot([ox, ex], [oy, ey], linewidth=4, color=link_color, alpha=alpha*0.8)
        ax.scatter([ex], [ey], s=100, color=link_color, zorder=10, edgecolors='black', alpha=alpha)
        
        # Plot link collision circles
        if link.circle_offsets.numel() > 0:
            for c_i in range(link.circle_offsets.shape[0]):
                cx, cy = circle_positions[i][0, c_i, :].cpu().numpy()
                circle = Circle((cx, cy), radius=link.circle_radii[c_i].item(),
                              fill=False, edgecolor=link_color, linestyle='--', alpha=alpha*0.6)
                ax.add_patch(circle)

def add_sample_configurations(robot, ax, config_grid, combined_mask, joint_ranges, resolution, tensor_args):
    """Add sample robot configurations to workspace plot."""
    # Find some collision-free configurations
    free_indices = torch.where(~combined_mask)[0]
    
    if len(free_indices) > 0:
        # Sample a few configurations
        sample_indices = free_indices[torch.randint(0, len(free_indices), (3,))]
        
        for i, idx in enumerate(sample_indices):
            config = config_grid[idx]
            
            # Set robot to this configuration
            robot.x = torch.cat([config, torch.zeros(len(config), **tensor_args)]).unsqueeze(0)
            
            # Get forward kinematics
            full_angles = robot._build_full_angle_vector(config.unsqueeze(0))
            origins, endpoints, circle_positions = robot.forward_kinematics(full_angles)
            
            # Plot with transparency
            alpha = 0.3
            sample_color = ['purple', 'cyan', 'yellow'][i]
            
            for j, link in enumerate(robot.links):
                ox, oy = origins[0, j, :].cpu().numpy()
                ex, ey = endpoints[0, j, :].cpu().numpy()
                ax.plot([ox, ex], [oy, ey], linewidth=2, color=sample_color, alpha=alpha)

def create_detailed_cspace_plots(robot, obstacle_masks, combined_mask, sdf_values,
                                joint_ranges, resolution):
    """Create separate detailed C-space plots."""
    
    # Individual obstacle contributions
    num_obstacles = len(robot.obstacles)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    extent = [joint_ranges[0][0], joint_ranges[0][1], joint_ranges[1][0], joint_ranges[1][1]]
    
    for obs_idx in range(min(num_obstacles, 6)):  # Limit to 6 for layout
        row = obs_idx // 3
        col = obs_idx % 3
        ax = axes[row, col]
        
        collision_grid = obstacle_masks[obs_idx].view(resolution, resolution).cpu().numpy()
        
        # Create binary colormap
        cmap = mcolors.ListedColormap(['white', robot.obstacles[obs_idx].color])
        
        ax.imshow(collision_grid.T, extent=extent, origin='lower', 
                 cmap=cmap, aspect='auto', alpha=0.8)
        ax.set_xlabel('Joint 1 (rad)')
        ax.set_ylabel('Joint 2 (rad)')
        ax.set_title(f'Obstacle {obs_idx + 1}: {robot.obstacles[obs_idx].color}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        collision_count = torch.sum(obstacle_masks[obs_idx]).item()
        total_configs = len(obstacle_masks[obs_idx])
        collision_rate = collision_count / total_configs * 100
        ax.text(0.02, 0.98, f'Collision: {collision_rate:.1f}%', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Hide unused subplots
    for obs_idx in range(num_obstacles, 6):
        row = obs_idx // 3
        col = obs_idx % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("cspace_per_obstacle.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Combined C-space with clearance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Combined collision map
    combined_grid = combined_mask.view(resolution, resolution).cpu().numpy()
    ax1.imshow(combined_grid.T, extent=extent, origin='lower', 
              cmap='RdYlGn_r', aspect='auto', alpha=0.8)
    ax1.set_xlabel('Joint 1 (rad)')
    ax1.set_ylabel('Joint 2 (rad)')
    ax1.set_title('Combined C-space (Red: Collision, Green: Free)')
    ax1.grid(True, alpha=0.3)
    
    # Clearance map
    sdf_plot = sdf_values.view(resolution, resolution).cpu().numpy()
    sdf_plot[combined_grid] = -0.1  # Mask collision regions
    
    im2 = ax2.imshow(sdf_plot.T, extent=extent, origin='lower', 
                    cmap='viridis', aspect='auto')
    ax2.set_xlabel('Joint 1 (rad)')
    ax2.set_ylabel('Joint 2 (rad)')
    ax2.set_title('Clearance Map')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax, label='Clearance Distance')
    
    plt.tight_layout()
    plt.savefig("cspace_combined.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Detailed C-space plots saved!")

if __name__ == "__main__":
    main()
