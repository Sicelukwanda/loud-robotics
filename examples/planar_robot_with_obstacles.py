#!/usr/bin/env python3
"""
Example script demonstrating planar robot with obstacles.

This script shows how to:
1. Create a planar robot with obstacles
2. Add circular and rectangular obstacles
3. Visualize the environment SDF
4. Show the robot in the environment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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
    XLIM = [-3, 3]
    YLIM = [-0.5, 3]

    print("Creating planar robot with obstacles...")
    
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
    
    # Test environment SDF computation
    print("\nTesting environment SDF computation...")
    test_points = torch.tensor([
        [0.0, 0.0],     # Origin
        [1.5, 0.5],     # Inside circle obstacle
        [2.0, 2.0],     # Free space
        [0.5, -1.5],    # Near rectangle obstacle
    ], **tensor_args)
    
    env_sdf = robot.environment_sdf_at_points(test_points)
    print("SDF values at test points:")
    for i, (point, sdf_val) in enumerate(zip(test_points, env_sdf)):
        print(f"  Point {point.numpy()}: SDF = {sdf_val.item():.3f}")
    
    # Visualization
    print("\nCreating visualizations...")
    
    # 1. Plot environment SDF
    print("Plotting environment SDF...")
    env_fig = robot.plot_environment_sdf(resolution=150, sdf_min=-0.5, sdf_max=0.5, 
                                        xlim=XLIM, ylim=YLIM)
    # env_fig.suptitle("Environment SDF with Obstacles", fontsize=16)
    env_fig.savefig("environment_sdf_with_obstacles.pdf", bbox_inches='tight', dpi=300)
    print("Saved environment SDF plot to: environment_sdf_with_obstacles.pdf")
    
    # 2. Plot robot state with obstacles
    print("Plotting robot state...")
    robot_fig = robot.plot_robot_state()
    
    # Add obstacles to robot state plot
    robot_ax = robot_fig.gca()
    for obstacle in robot.obstacles:
        if isinstance(obstacle, CircleObstacle):
            center = obstacle.center.cpu().numpy()
            radius = obstacle.radius.item()
            from matplotlib.patches import Circle
            circle = Circle(center, radius, fill=True, color='red', alpha=0.3, edgecolor='darkred')
            robot_ax.add_patch(circle)
        elif isinstance(obstacle, RectangleObstacle):
            center = obstacle.center.cpu().numpy()
            width = obstacle.width.item()
            height = obstacle.height.item()
            angle_deg = np.degrees(obstacle.angle.item())
            from matplotlib.patches import Rectangle
            
            # Use precomputed rotated bottom-left corner from obstacle
            bottom_left = obstacle.get_matplotlib_bottom_left()
            
            rect = Rectangle(bottom_left, width, height, angle=angle_deg, 
                           fill=True, color='red', alpha=0.3, edgecolor='darkred')
            robot_ax.add_patch(rect)
    robot_ax.set_xlim(XLIM[0], XLIM[1])
    robot_ax.set_ylim(YLIM[0], YLIM[1])
    # robot_fig.suptitle("Robot Configuration with Obstacles", fontsize=16)
    robot_fig.savefig("robot_configuration_with_obstacles.pdf", bbox_inches='tight', dpi=300)
    print("Saved robot configuration plot to: robot_configuration_with_obstacles.pdf")
    
    # 3. Demonstrate robot motion with obstacles
    print("Simulating robot motion...")
    
    # Create animation-like visualization
    motion_fig, motion_ax = plt.subplots(1, 1, figsize=(10, 8))
    # motion_ax.set_title("Robot Motion with Obstacles")
    motion_ax.set_xlabel("X")
    motion_ax.set_ylabel("Y")
    motion_ax.set_aspect('equal')
    
    
    # Set plot limits
    arm_length = sum([l.length for l in links]) * 1.2
    motion_ax.set_xlim(XLIM[0], XLIM[1])
    motion_ax.set_ylim(YLIM[0], YLIM[1])
    motion_ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            center = obstacle.center.cpu().numpy()
            radius = obstacle.radius.item()
            circle = plt.Circle(center, radius, fill=True, color='red', alpha=0.3, edgecolor='darkred')
            motion_ax.add_patch(circle)
        elif isinstance(obstacle, RectangleObstacle):
            center = obstacle.center.cpu().numpy()
            width = obstacle.width.item()
            height = obstacle.height.item()
            angle_deg = np.degrees(obstacle.angle.item())
            
            # Use precomputed rotated bottom-left corner from obstacle
            bottom_left = obstacle.get_matplotlib_bottom_left()
            
            rect = plt.Rectangle(bottom_left, width, height, angle=angle_deg, 
                               fill=True, color='red', alpha=0.3, edgecolor='darkred')
            motion_ax.add_patch(rect)
    
    # Simulate some robot configurations
    angles_sequence = [
        torch.tensor([0.3, -0.5, 0.8], **tensor_args),
        torch.tensor([0.8, -0.3, 0.5], **tensor_args),
        torch.tensor([1.2, 0.2, 0.2], **tensor_args),
        torch.tensor([0.5, 0.8, -0.3], **tensor_args),
    ]
    
    colors = ['blue', 'green', 'orange', 'purple']
    
    for i, angles in enumerate(angles_sequence):
        # Set robot configuration
        robot.x = torch.cat([angles, torch.zeros(len(angles), **tensor_args)]).unsqueeze(0)
        
        # Get forward kinematics
        full_angles = robot._build_full_angle_vector(angles.unsqueeze(0))
        origins, endpoints, circle_positions = robot.forward_kinematics(full_angles)
        
        # Plot robot configuration
        particle_color = colors[i]
        alpha = 0.7 - i * 0.15  # Fade older configurations
        
        for j, link in enumerate(links):
            ox, oy = origins[0, j, :].cpu().numpy()
            ex, ey = endpoints[0, j, :].cpu().numpy()
            motion_ax.plot([ox, ex], [oy, ey], linewidth=3, color=particle_color, alpha=alpha,
                          label=f'Config {i+1}' if j == 0 else '')
            motion_ax.scatter([ex], [ey], s=50, color=particle_color, alpha=alpha, zorder=10)
            
            # Plot link circles
            if link.circle_offsets.numel() > 0:
                for c_i in range(link.circle_offsets.shape[0]):
                    cx, cy = circle_positions[j][0, c_i, :].cpu().numpy()
                    circle = plt.Circle((cx, cy), radius=link.circle_radii[c_i].item(),
                                      fill=False, edgecolor=particle_color, linestyle='--', alpha=alpha)
                    motion_ax.add_patch(circle)
    
    motion_ax.legend()
    motion_fig.savefig("robot_motion_with_obstacles.pdf", bbox_inches='tight', dpi=300)
    print("Saved robot motion plot to: robot_motion_with_obstacles.pdf")
    
    # 4. Test adding and removing obstacles dynamically
    print("\nTesting dynamic obstacle management...")
    
    # Add a new obstacle
    new_obstacle = CircleObstacle(center=(0.0, 2.0), radius=0.25, tensor_args=tensor_args)
    robot.add_obstacle(new_obstacle)
    print(f"Added obstacle. Total obstacles: {len(robot.obstacles)}")
    
    # Compute SDF again
    env_sdf_new = robot.environment_sdf_at_points(test_points)
    print("SDF values after adding obstacle:")
    for i, (point, sdf_val) in enumerate(zip(test_points, env_sdf_new)):
        print(f"  Point {point.numpy()}: SDF = {sdf_val.item():.3f}")
    
    # Clear all obstacles
    robot.clear_obstacles()
    print(f"Cleared obstacles. Total obstacles: {len(robot.obstacles)}")
    
    # Show all plots
    plt.show()
    print("\nPDF files saved:")
    print("  - environment_sdf_with_obstacles.pdf")
    print("  - robot_configuration_with_obstacles.pdf") 
    print("  - robot_motion_with_obstacles.pdf")
    print("\nExample completed!")

if __name__ == "__main__":
    main()
