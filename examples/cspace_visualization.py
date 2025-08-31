#!/usr/bin/env python3
"""
Configuration Space Visualization Script

This script demonstrates:
1. Computing the configuration space for a 2DOF planar robot with obstacles
2. Visualizing each obstacle's contribution to C-space with different colors overlaid
3. Joint 0 on horizontal axis, Joint 1 on vertical axis
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend explicitly
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    
    print("Creating 2DOF planar robot for C-space analysis...")
    
    # Create robot links
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
        ),
        Link(
            length=0.8, 
            circle_offsets=[0.15, 0.4, 0.65], 
            circle_radii=[0.15, 0.15, 0.15],
            tensor_args=tensor_args,
        )
    ]
    
    # Create obstacles
    obstacles = [
        # Circular obstacles
        # CircleObstacle(center=(1.5, 0.5), radius=0.3, tensor_args=tensor_args),
        # CircleObstacle(center=(-1.0, 1.2), radius=0.4, tensor_args=tensor_args),
        
        # Rectangular obstacles
        RectangleObstacle(center=(1.0, 1.0), width=1.3, height=0.1, 
                         angle=0.0, tensor_args=tensor_args),
        RectangleObstacle(center=(1.0, 0.475), width=0.1, height=0.95, 
                         angle=0.0, tensor_args=tensor_args),
        RectangleObstacle(center=(0, -10.0), width=20, height=20.0, 
                         tensor_args=tensor_args),
    ]
    
    # Initialize robot
    robot = DPlanarRobot(
        links=links,
        obstacles=obstacles,
        starting_angle_config=torch.tensor([np.pi/2.0, -0.5, 0.8], **tensor_args),
        tensor_args=tensor_args,
        seed=42
    )
    
    print(f"Robot created with {len(robot.obstacles)} obstacles")
    robot.reset(num_particles=1)
    
    # Joint ranges for 2DOF robot
    joint_ranges = [(-np.pi, np.pi), (-np.pi, np.pi)]  # Joint 0, Joint 1
    
    print(f"Computing C-space for 2DOF robot...")
    print(f"Joint ranges: {joint_ranges}")
    print(f"Axis layout: Joint 0 (horizontal), Joint 1 (vertical)")
    
    # Compute C-space
    resolution = 50
    print(f"Using resolution: {resolution}x{resolution}")
    
    create_per_obstacle_cspace_plot(robot, joint_ranges, resolution, tensor_args)


def test_configuration(robot, config, tensor_args):
    """
    Test a single configuration for collision.
    Similar to JavaScript isValidConfiguration function.
    
    Args:
        robot: DPlanarRobot instance
        config: torch.tensor of joint angles [joint0, joint1, ...]
        tensor_args: tensor arguments
        
    Returns:
        bool: True if collision detected, False otherwise
    """
    # Store original state
    original_x = robot.x.clone()
    original_particles = robot.num_particles
    
    try:
        # Create robot state for this configuration
        # The robot expects [positions, velocities] for actuated joints
        batch_size = 1
        
        # Create full configuration (including fixed joints)
        full_config = torch.zeros(len(robot.links), **tensor_args)
        for i, joint_idx in enumerate(robot.actuated_indices):
            if i < len(config):
                full_config[joint_idx] = config[i]
        
        # Create state: [actuated_positions, actuated_velocities]
        actuated_positions = full_config[robot.actuated_indices].unsqueeze(0)  # (1, num_actuated)
        actuated_velocities = torch.zeros_like(actuated_positions)
        robot_state = torch.cat([actuated_positions, actuated_velocities], dim=1)
        
        # Set robot state
        robot.x = robot_state
        robot.num_particles = 1
        
        # Simple collision detection using workspace sampling
        if len(robot.obstacles) == 0:
            return False
            
        # Create test points for collision checking
        arm_length = sum([l.length for l in robot.links])
        test_points = robot._create_workspace_test_points(arm_length, resolution=20, adaptive=False)
        
        # Get robot and environment SDFs
        robot_sdf = robot.sdf_at_points(test_points)  # (1, N)
        env_sdf = robot.environment_sdf_at_points(test_points)  # (N,)
        
        # Collision if robot and environment overlap
        robot_inside = robot_sdf[0] < 0  # (N,)
        env_inside = env_sdf < 0  # (N,)
        collision_points = robot_inside & env_inside
        
        return torch.any(collision_points).item()
        
    finally:
        # Restore original state
        robot.x = original_x
        robot.num_particles = original_particles


def create_per_obstacle_cspace_plot(robot, joint_ranges, resolution, tensor_args):
    """
    Create a plot showing per-obstacle C-space contributions using incremental rendering.
    Based on the JavaScript approach - renders each obstacle separately and overlays them.
    Joint 0 on horizontal axis, Joint 1 on vertical axis.
    """
    print("Computing per-obstacle C-space contributions (JavaScript-style incremental rendering)...")
    
    # Create joint value grids
    joint0_vals = torch.linspace(joint_ranges[0][0], joint_ranges[0][1], resolution, **tensor_args)
    joint1_vals = torch.linspace(joint_ranges[1][0], joint_ranges[1][1], resolution, **tensor_args)
    
    print(f"Resolution: {resolution}x{resolution} = {resolution*resolution:,} configurations")
    
    # Initialize RGB image data (like the JavaScript imgData)
    img_data = np.zeros((resolution, resolution, 4), dtype=np.uint8)  # RGBA
    
    # Define colors for obstacles (similar to JavaScript colors array)
    color_palette = [
        [255, 0, 0],      # red
        [255, 165, 0],    # orange  
        [0, 128, 0],      # green
        [0, 0, 255],      # blue
        [128, 0, 128],    # purple
        [128, 128, 128]   # gray
    ]
    
    print(f"Processing {len(robot.obstacles)} obstacles...")
    
    # Create figure early to show intermediate updates (like JavaScript)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    extent = [joint_ranges[0][0], joint_ranges[0][1], joint_ranges[1][0], joint_ranges[1][1]]
    
    # Process each obstacle incrementally (like JavaScript iteration())
    for obs_idx, obstacle in enumerate(robot.obstacles):
        print(f"\nRendering obstacle {obs_idx + 1}/{len(robot.obstacles)}...")
        
        # Get color for this obstacle (consistent with JavaScript color palette)
        color_rgb = color_palette[obs_idx % len(color_palette)]
        r, g, b = color_rgb
        
        print(f"  Using color: RGB({r}, {g}, {b})")
        
        # Create temporary robot with only this obstacle
        temp_robot = DPlanarRobot(
            links=robot.links,
            obstacles=[obstacle],
            tensor_args=tensor_args,
            seed=42
        )
        temp_robot.reset(num_particles=1)
        
        collision_count = 0
        update_frequency = max(1, resolution // 10)  # Update every 10% of progress
        
        # Render this obstacle's C-space contribution
        # Following JavaScript logic: for each joint configuration
        for ai in range(resolution):  # Joint 0 (angle a)
            if ai % update_frequency == 0:  # Progress reporting
                progress = (ai / resolution) * 100
                print(f"  Progress: {progress:.1f}%", end='', flush=True)
                if ai > 0:
                    print(f" ({collision_count} collisions so far)")
                else:
                    print()
            
            angle_a = joint0_vals[ai].item()
            
            for bi in range(resolution):  # Joint 1 (angle b) 
                angle_b = joint1_vals[bi].item()
                
                # Test this configuration
                config = torch.tensor([angle_a, angle_b], **tensor_args)
                collision = test_configuration(temp_robot, config, tensor_args)
                
                if collision:
                    collision_count += 1
                    # Set pixel with color blending (like JavaScript setPixel())
                    # Joint 0 on horizontal axis (X), Joint 1 on vertical axis (Y)
                    pixel_offset_y = bi  # Joint 1 on vertical axis
                    pixel_offset_x = ai  # Joint 0 on horizontal axis
                    
                    # Alpha blending: if pixel already has color, blend with new color
                    current_alpha = img_data[pixel_offset_y, pixel_offset_x, 3]
                    if current_alpha > 0:
                        # Blend colors (average like in JavaScript)
                        img_data[pixel_offset_y, pixel_offset_x, 0] = (r + img_data[pixel_offset_y, pixel_offset_x, 0]) // 2
                        img_data[pixel_offset_y, pixel_offset_x, 1] = (g + img_data[pixel_offset_y, pixel_offset_x, 1]) // 2
                        img_data[pixel_offset_y, pixel_offset_x, 2] = (b + img_data[pixel_offset_y, pixel_offset_x, 2]) // 2
                    else:
                        # Set new color
                        img_data[pixel_offset_y, pixel_offset_x, 0] = r
                        img_data[pixel_offset_y, pixel_offset_x, 1] = g
                        img_data[pixel_offset_y, pixel_offset_x, 2] = b
                        img_data[pixel_offset_y, pixel_offset_x, 3] = 255
        
        collision_rate = collision_count / (resolution * resolution) * 100
        print(f"  → Obstacle {obs_idx + 1} final: {collision_count:,} collisions ({collision_rate:.1f}%)")
        
        # Show intermediate result (like JavaScript putImageData + drawX)
        ax.clear()
        ax.imshow(img_data, extent=extent, origin='lower', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Joint 0 (rad)', fontsize=12)
        ax.set_ylabel('Joint 1 (rad)', fontsize=12)
        ax.set_title(f'C-Space Rendering Progress: {obs_idx + 1}/{len(robot.obstacles)} obstacles', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add current robot configuration marker (like JavaScript drawX())
        current_joint0 = robot.x[0, 0].item()
        current_joint1 = robot.x[0, 1].item()
        ax.plot(current_joint0, current_joint1, 'w+', markersize=10, markeredgewidth=2)
        ax.plot(current_joint0, current_joint1, 'k+', markersize=8, markeredgewidth=1)
    
    # Create the final plot and display the complete image
    ax.clear()  # Clear the intermediate plot
    ax.imshow(img_data, extent=extent, origin='lower', aspect='auto', interpolation='nearest')
    
    # Formatting
    ax.set_xlabel('Joint 0 (rad)', fontsize=12)
    ax.set_ylabel('Joint 1 (rad)', fontsize=12)
    # ax.set_title('Per-Obstacle Configuration Space\n(Incremental Rendering with Color Blending)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend showing obstacle colors
    legend_elements = []
    obstacle_labels = []
    for obs_idx in range(len(robot.obstacles)):
        color_rgb = color_palette[obs_idx % len(color_palette)]
        color_norm = [c/255.0 for c in color_rgb]  # Normalize to [0,1] for matplotlib
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color_norm, 
                                           edgecolor='black', linewidth=1))
        obstacle_labels.append(f'Obstacle {obs_idx + 1}')
    
    if legend_elements:
        ax.legend(legend_elements, obstacle_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add current robot configuration marker (like JavaScript drawX())
    current_joint0 = robot.x[0, 0].item()  # Current joint 0 angle
    current_joint1 = robot.x[0, 1].item()  # Current joint 1 angle
    
    # Draw crosshair marker
    ax.plot(current_joint0, current_joint1, 'w+', markersize=10, markeredgewidth=2, 
            label='Current Configuration')
    ax.plot(current_joint0, current_joint1, 'k+', markersize=8, markeredgewidth=1)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = "cspace_per_obstacle_overlay.pdf"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as: {output_filename}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    print("\n✓ Per-obstacle C-space visualization completed!")
    print(f"Saved as: {output_filename}")


def evaluate_configurations_simple(robot, config_grid, tensor_args):
    """
    Simple collision evaluation for configurations.
    """
    total_configs = config_grid.shape[0]
    batch_size = 1000
    collision_mask = torch.zeros(total_configs, dtype=torch.bool, device=tensor_args['device'])
    
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
        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            progress = (end_idx / total_configs) * 100
            print(f"    Progress: {progress:.1f}%")
        
        # Create full robot configurations (including fixed joint)
        full_configs = torch.zeros(batch_size_actual, len(robot.links), device=tensor_args['device'], dtype=tensor_args['dtype'])
        
        # Set actuated joint values (skip the fixed first joint)
        for i, joint_idx in enumerate(robot.actuated_indices):
            if i < batch_configs.shape[1]:
                full_configs[:, joint_idx] = batch_configs[:, i]
        
        # Create robot state (position and velocity)
        batch_states = torch.cat([
            full_configs,
            torch.zeros(batch_size_actual, len(robot.links), device=tensor_args['device'], dtype=tensor_args['dtype'])
        ], dim=1)
        
        # Set robot state
        robot.x = batch_states
        robot.num_particles = batch_size_actual
        
        # Simple collision detection using robot's built-in methods
        if len(robot.obstacles) > 0:
            # Create test points for collision checking
            arm_length = sum([l.length for l in robot.links])
            test_points = robot._create_workspace_test_points(arm_length, resolution=30, adaptive=True)
            
            # Get robot and environment SDFs
            robot_sdf = robot.sdf_at_points(test_points)
            env_sdf = robot.environment_sdf_at_points(test_points)
            
            # Collision if robot and environment overlap
            robot_inside = robot_sdf < 0
            env_inside = env_sdf < 0
            collision_points = robot_inside & env_inside.unsqueeze(0)
            batch_collisions = torch.any(collision_points, dim=1)
            
            collision_mask[start_idx:end_idx] = batch_collisions
    
    # Restore original robot state
    robot.x = original_x
    robot.num_particles = original_particles
    
    return collision_mask


if __name__ == "__main__":
    main()
