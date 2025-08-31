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
    resolution = 500
    print(f"Using resolution: {resolution}x{resolution}")
    
    create_per_obstacle_cspace_plot(robot, joint_ranges, resolution, tensor_args)


def create_per_obstacle_cspace_plot(robot, joint_ranges, resolution, tensor_args):
    """
    Create a plot showing per-obstacle C-space contributions using the optimized method.
    Uses robot's built-in compute_configuration_space_per_obstacle for speed.
    Joint 0 on horizontal axis, Joint 1 on vertical axis.
    """
    print("Computing per-obstacle C-space contributions using optimized method...")
    
    # Use the robot's optimized method directly
    config_grid, obstacle_masks, combined_mask, sdf_values = robot.compute_configuration_space_per_obstacle(
        joint_ranges, resolution
    )
    
    print(f"Resolution: {resolution}x{resolution} = {config_grid.shape[0]:,} configurations")
    
    # Initialize RGB image data for visualization
    img_data = np.zeros((resolution, resolution, 4), dtype=np.uint8)  # RGBA
    
    # Define colors for obstacles (matching JavaScript colors)
    color_palette = [
        [255, 0, 0],      # red
        [255, 165, 0],    # orange  
        [0, 128, 0],      # green
        [0, 0, 255],      # blue
        [128, 0, 128],    # purple
        [128, 128, 128]   # gray
    ]
    
    print(f"Creating visualization for {len(robot.obstacles)} obstacles...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    extent = [joint_ranges[0][0], joint_ranges[0][1], joint_ranges[1][0], joint_ranges[1][1]]
    
    # Apply each obstacle's mask to the image with color blending
    for obs_idx, mask in enumerate(obstacle_masks):
        color_rgb = color_palette[obs_idx % len(color_palette)]
        r, g, b = color_rgb
        
        print(f"  Applying obstacle {obs_idx + 1} with color RGB({r}, {g}, {b})")
        
        # Reshape mask to 2D grid
        mask_2d = mask.view(resolution, resolution).cpu().numpy()
        
        collision_count = np.sum(mask_2d)
        collision_rate = collision_count / (resolution * resolution) * 100
        print(f"    → {collision_count:,} collisions ({collision_rate:.1f}%)")
        
        # Apply color blending for collision regions
        for i in range(resolution):
            for j in range(resolution):
                if mask_2d[i, j]:  # Collision detected
                    # Joint 0 on horizontal (X), Joint 1 on vertical (Y)
                    pixel_y = j  # Joint 1 on vertical axis
                    pixel_x = i  # Joint 0 on horizontal axis
                    
                    # Alpha blending like JavaScript
                    current_alpha = img_data[pixel_y, pixel_x, 3]
                    if current_alpha > 0:
                        # Blend colors (average)
                        img_data[pixel_y, pixel_x, 0] = (r + img_data[pixel_y, pixel_x, 0]) // 2
                        img_data[pixel_y, pixel_x, 1] = (g + img_data[pixel_y, pixel_x, 1]) // 2
                        img_data[pixel_y, pixel_x, 2] = (b + img_data[pixel_y, pixel_x, 2]) // 2
                    else:
                        # Set new color
                        img_data[pixel_y, pixel_x, 0] = r
                        img_data[pixel_y, pixel_x, 1] = g
                        img_data[pixel_y, pixel_x, 2] = b
                        img_data[pixel_y, pixel_x, 3] = 255
    
    # Display the final image
    ax.imshow(img_data, extent=extent, origin='lower', aspect='auto', interpolation='nearest')
    
    # Formatting
    ax.set_xlabel('Joint 0 (rad)', fontsize=12)
    ax.set_ylabel('Joint 1 (rad)', fontsize=12)
    ax.set_title('Per-Obstacle Configuration Space\n(Optimized Direct Collision Detection)', fontsize=14)
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
    
    # Add current robot configuration marker
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
    
    print("\n✓ Optimized per-obstacle C-space visualization completed!")
    print(f"Saved as: {output_filename}")


if __name__ == "__main__":
    main()
