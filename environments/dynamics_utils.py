import numpy as np
import torch

def trajectory_to_transitions(X, state_dim, action_dim):
    """
    Given a batch of trajectories with shape (num_trajectories, horizon, state_dim + action_dim),
    convert trajectories to transitions with shape (num_transitions, state_dim + action_dim + state_dim).
    """
    num_trajectories, horizon, _ = X.shape
    num_transitions = num_trajectories * (horizon - 1)
    
    transitions = np.zeros((num_transitions, state_dim + action_dim + state_dim))
    
    transitions_list = []
    for i in range(num_trajectories):
        diffs = X[i, 1:, :state_dim] - X[i, :-1, :state_dim]
        transitions_i = np.concatenate([X[i, :-1, :], diffs], axis=1)
        transitions_list.append(transitions_i)

    transitions = np.concatenate(transitions_list, axis=0)
    return transitions

def circle_sdf(point_positions, circle_centers, circle_radii):
    """
    Compute the SDF of multiple points w.r.t. a set of circles.
    Autograd-friendly.

    Inputs:
        point_positions: (N, 2) tensor, positions of query points
        circle_centers: (M, 2) tensor, positions of circle centers
        circle_radii: (M,) tensor of radii for each circle
    
    Returns:
        sdf_values: (N,) tensor, SDF for each point w.r.t the union of the given circles.
                    The SDF of a union of circles is min(distance_to_each_circle - radius).
    """
    # point_positions: (N, 2)
    # circle_centers: (M, 2)
    # We want to compute distance from each point to each circle center
    # Expand and broadcast:
    # distances: shape (N, M)
    # distances[i, j] = distance from point i to circle j
    diff = point_positions.unsqueeze(1) - circle_centers.unsqueeze(0)  # (N, M, 2)
    distances = torch.sqrt((diff**2).sum(dim=2))  # (N, M)

    # For each circle, SDF contribution at a point = distance - radius
    sdf_per_circle = distances - circle_radii.unsqueeze(0)  # (N, M)

    # The SDF of the union is the minimum over all circles
    sdf_values, _ = torch.min(sdf_per_circle, dim=1)  # (N,)

    return sdf_values