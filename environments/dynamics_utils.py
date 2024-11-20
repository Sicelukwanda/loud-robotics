import numpy as np

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
