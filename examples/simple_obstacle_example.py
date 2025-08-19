#!/usr/bin/env python3
"""
Simple example showing planar robot with obstacle SDF computation.

This demonstrates the core obstacle functionality without complex visualization.
"""

import torch
import numpy as np

from loud_robotics.environments.planar_robot import (
    DPlanarRobot, 
    Link, 
    CircleObstacle, 
    RectangleObstacle
)

def main():
    # Set up device and random seed
    tensor_args = {'dtype': torch.float32, 'device': 'cpu'}
    
    print("=== Planar Robot with Obstacles - Simple Example ===\n")
    
    # Create a simple 2-link robot
    links = [
        Link(length=1.0, circle_offsets=[0.5], circle_radii=[0.1], tensor_args=tensor_args),
        Link(length=0.8, circle_offsets=[0.4], circle_radii=[0.08], tensor_args=tensor_args)
    ]
    
    # Create obstacles
    obstacles = [
        CircleObstacle(center=(1.2, 0.3), radius=0.2, tensor_args=tensor_args),
        RectangleObstacle(center=(-0.5, -1.0), width=0.6, height=0.4, tensor_args=tensor_args)
    ]
    
    # Store original obstacles for later restoration
    original_obstacles = obstacles.copy()
    
    # Initialize robot with obstacles
    robot = DPlanarRobot(
        links=links,
        obstacles=obstacles,
        starting_angle_config=torch.tensor([0.5, -0.3], **tensor_args),
        tensor_args=tensor_args
    )
    
    print(f"Created robot with {len(robot.obstacles)} obstacles:")
    print(f"  - Circle obstacle at (1.2, 0.3) with radius 0.2")
    print(f"  - Rectangle obstacle at (-0.5, -1.0) with size 0.6x0.4\n")
    
    # Reset robot
    robot.reset(num_particles=1)
    
    # Test SDF computation at various points
    test_points = torch.tensor([
        [0.0, 0.0],      # Origin (free space)
        [1.2, 0.3],      # Center of circle obstacle
        [1.0, 0.3],      # Near circle obstacle
        [-0.5, -1.0],    # Center of rectangle obstacle
        [-0.2, -1.0],    # Near rectangle obstacle
        [2.0, 2.0],      # Far from obstacles
    ], **tensor_args)
    
    # Compute environment SDF (obstacles only)
    env_sdf = robot.environment_sdf_at_points(test_points)
    
    print("Environment SDF values (obstacles only):")
    print("Point\t\t\tSDF Value\tInterpretation")
    print("-" * 60)
    
    for i, (point, sdf_val) in enumerate(zip(test_points, env_sdf)):
        x, y = point.numpy()
        sdf = sdf_val.item()
        
        if sdf < -0.01:
            interpretation = "Inside obstacle"
        elif sdf < 0.01:
            interpretation = "On obstacle boundary"
        else:
            interpretation = f"Free space ({sdf:.3f}m from obstacle)"
        
        print(f"({x:5.1f}, {y:5.1f})\t\t{sdf:8.3f}\t{interpretation}")
    
    print("\n" + "="*60)
    
    # Test robot SDF
    robot_sdf = robot.sdf_at_points(test_points)
    print(f"\nRobot SDF values (for current configuration):")
    print("Point\t\t\tSDF Value\tInterpretation")
    print("-" * 60)
    
    for i, (point, sdf_val) in enumerate(zip(test_points, robot_sdf[0])):  # robot_sdf[0] for first particle
        x, y = point.numpy()
        sdf = sdf_val.item()
        
        if sdf < -0.01:
            interpretation = "Inside robot"
        elif sdf < 0.01:
            interpretation = "On robot boundary"
        else:
            interpretation = f"Free space ({sdf:.3f}m from robot)"
        
        print(f"({x:5.1f}, {y:5.1f})\t\t{sdf:8.3f}\t{interpretation}")
    
    # Demonstrate dynamic obstacle management
    print(f"\n=== Dynamic Obstacle Management ===")
    print(f"Current obstacles: {len(robot.obstacles)}")
    
    # Add another obstacle
    new_obstacle = CircleObstacle(center=(0.0, 1.5), radius=0.15, tensor_args=tensor_args)
    robot.add_obstacle(new_obstacle)
    print(f"Added circle obstacle at (0.0, 1.5). Total obstacles: {len(robot.obstacles)}")
    
    # Test SDF at a point near the new obstacle
    test_point_near_new = torch.tensor([[0.0, 1.4]], **tensor_args)
    sdf_near_new = robot.environment_sdf_at_points(test_point_near_new)
    print(f"SDF at (0.0, 1.4) near new obstacle: {sdf_near_new.item():.3f}")
    
    # Clear all obstacles
    robot.clear_obstacles()
    print(f"Cleared all obstacles. Total obstacles: {len(robot.obstacles)}")
    
    # Test SDF with no obstacles
    sdf_no_obstacles = robot.environment_sdf_at_points(test_points[:3])
    print("SDF values with no obstacles (should be large positive):")
    for i, (point, sdf_val) in enumerate(zip(test_points[:3], sdf_no_obstacles)):
        x, y = point.numpy()
        sdf = sdf_val.item()
        print(f"  ({x:5.1f}, {y:5.1f}): {sdf}")
    
    # Create a simple visualization and save as PDF
    print("\nCreating visualization...")
    
    # Reset robot with obstacles for visualization
    robot.clear_obstacles()  # Clear first to be safe
    for obstacle in original_obstacles:  # Re-add original obstacles properly
        robot.add_obstacle(obstacle)
    
    print(f"Restored {len(robot.obstacles)} obstacles for visualization")
    
    # Create environment SDF plot
    env_fig = robot.plot_environment_sdf(resolution=100, sdf_min=-0.5, sdf_max=0.5)
    env_fig.suptitle("Simple Obstacle Example - Environment SDF", fontsize=14)
    env_fig.savefig("simple_obstacle_example.pdf", bbox_inches='tight', dpi=300)
    print("Saved visualization to: simple_obstacle_example.pdf")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
