import torch
from torch.distributions import MultivariateNormal
from matplotlib import pyplot as plt

import numpy as np
import GPy

from environments import InvertedPendulum
from environments import trajectory_to_transitions

from models import IncrementalGPList, GPList
from models.utils import plot_gp


def unroll_forward(init_state, dynamics_model, action_sequence):
    """Unroll the model from a given initial state for a given sequence of actions."""
    states = [init_state]
    for a in action_sequence:
        s = dynamics_model(np.concatenate([states[-1], a]))
        states.append(s)
    return states


def simulate(
    env, num_particles, horizon, tensor_args, seed, visualize=False, plot=False
):

    state = env.reset(num_particles=num_particles)
    # Run simulation steps with a dummy action (0's - for passive dynamics)

    # TODO: make the actions vector an arg
    dummy_action = torch.zeros((env.num_particles, env.action_dim), **tensor_args)

    trajectories = torch.zeros(
        (num_particles, horizon, env.state_dim + env.action_dim), **tensor_args
    )
    for i in range(horizon):
        # concatenate state and action
        aug_state = torch.cat([state, dummy_action], dim=1)
        trajectories[:, i, :] = aug_state
        state = env.step(dummy_action)
        print(f"Time elapsed: {i * env.dt:.2f} seconds, State: {env.x}")
        # print(f"State shape: {env.x.shape}")

        # Visualize final state
        if visualize:
            env.visualize()

    # Keep the plot open after the loop finishes
    plt.ioff()

    print("Trajectories shape:", trajectories.shape)

    if plot:
        # plot trajectories
        plt.figure()
        for i in range(num_particles):
            plt.plot(
                trajectories[i, :, 0].cpu().numpy(),
                label=f"Particle {i} $\\theta$",
                color=env.colors[i],
            )
            plt.plot(
                trajectories[i, :, 1].cpu().numpy(),
                "--",
                label=f"Particle {i} $\dot{{\\theta}}$",
                color=env.colors[i],
            )
        plt.legend()

    return trajectories


def main():

    # collect initial data
    # default tensor arguments
    tensor_args = {
        "dtype": torch.float32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # random seed
    seed = 0

    # environment
    env = InvertedPendulum(
        tensor_args=tensor_args, dt=0.05, seed=seed
    )  # default dt=0.07

    train_particles = 5
    test_particles = 1
    horizon = 50

    train_trajectories = simulate(
        env, train_particles, horizon, tensor_args, seed, visualize=False, plot=False
    )

    # maybe set new start state
    env.start_state = torch.tensor([torch.pi / 2.0, 0.0], **tensor_args)

    test_trajectories = simulate(
        env, test_particles, horizon, tensor_args, seed, visualize=False, plot=False
    )

    # train test split (over entire trajectories) - test with last trajectory
    # get transitions (state, action, state_diffs)
    train_transitions = trajectory_to_transitions(
        train_trajectories.cpu().numpy(), env.state_dim, env.action_dim
    )

    # TODO: Re-simulate but from different state for test trajectories
    test_transitions = trajectory_to_transitions(
        test_trajectories.cpu().numpy(), env.state_dim, env.action_dim
    )

    print("Train transitions shape:", train_transitions.shape)
    print("Test transitions shape:", test_transitions.shape)

    train_X = (
        torch.tensor(train_transitions[:, : -env.state_dim], **tensor_args)
        .cpu()
        .numpy()
    )
    train_Y = (
        torch.tensor(train_transitions[:, -env.state_dim :], **tensor_args)
        .cpu()
        .numpy()
    )

    test_X = (
        torch.tensor(test_transitions[:, : -env.state_dim], **tensor_args).cpu().numpy()
    )
    test_Y = (
        torch.tensor(test_transitions[:, -env.state_dim :], **tensor_args).cpu().numpy()
    )

    print("Inputs: X_train ", train_X.shape, ", X_test", test_X.shape)
    print("Targets Y_train ", train_Y.shape, ", Y_test", test_Y.shape)

    # build models

    # 1. GP
    # Reset the random seed before igp_list to ensure consistent RNG state
    np.random.seed(0)
    # Create a list of kernels, one for each output dimension
    kernel_list = [
        GPy.kern.RBF(input_dim=train_X.shape[1]) for _ in range(train_Y.shape[1])
    ]

    # Initialize the GPList
    gp_list = GPList(train_X, train_Y, kernel_list)

    # Optimize the GP models
    gp_list.optimize(messages=True)

    mu, var = gp_list.predict(test_X)
    plot_gp(test_X, mu, var, training_points=(test_X, test_Y))

    # 2. IGP
    # Reset the random seed before igp_list to ensure consistent RNG state
    np.random.seed(0)

    # Initialize the IncrementalGPList
    kernel_list_igp = [
        GPy.kern.RBF(input_dim=train_X.shape[1]) for _ in range(train_Y.shape[1])
    ]

    noise_variance = 0.0
    igp_list = IncrementalGPList(
        train_X,
        train_Y,
        kernel_list=kernel_list_igp,
        noise_var=noise_variance,
        reoptimize=True,
    )

    # # Optimize the IGP models
    # for i in range(test_Y.shape[1]):
    #     igp_list[i].optimize(messages=True) # not needed

    # Predict outputs without updating the models
    mu, cov = igp_list.predict_X(test_X, full_cov=False)

    # Plot the base GP predictions for the all output dimensions
    plt.figure()
    plot_gp(test_X, mu, cov, training_points=(test_X, test_Y))

    # trajectory sampling:

    num_paths = 3
    dummy_action = np.zeros((1, 1))

    plt.figure()
    for k in range(num_paths):
        igp_list.reset_sampling_gp()

        # xs_gp = test_X[:1]
        # xs_igp = test_X[:1]

        igp_states = [test_X[0:1, :-1]]
        gp_states = [test_X[0:1, :-1]]
        times = [0]
        # true_states = [test_X[0:1,:-1]]

        for i in range(1, horizon - 1):

            igp_aug_state = np.concatenate([igp_states[i - 1], dummy_action], axis=1)
            igp_ys = igp_list.predict_xs(igp_aug_state)  # This updates the sampling gps

            igp_states.append(igp_states[i - 1] + igp_ys)  # prev_state + state_diff

            # print(f"Sample {i+1}: Input {xs.flatten()}, Predicted Output {ys.flatten()}")

            # plt.clf()  # get current figure
            # m_base, C_base = igp_list.predict_X(X_new, full_cov=False)
            # plot_gp(X_new, m_base, C_base, training_points=(X, Y))

            gp_aug_state = np.concatenate([gp_states[i - 1], dummy_action], axis=1)
            mu, variance = gp_list.predict(
                gp_aug_state, full_cov=False
            )  # This updates the sampling gps

            # sample from predictive posterior
            gp_ys = np.random.multivariate_normal(
                mean=mu.flatten(), cov=np.diag(variance.flatten()), size=1
            )

            gp_states.append(gp_states[i - 1] + gp_ys)  # prev_state + state_diff

            times.append(i * env.dt)

        gp_arr = np.concatenate(gp_states, axis=0)
        # _ = plt.plot(times, gp_arr[:,0], "r", linewidth=2,  markersize=8, label="gp mean $\\theta$")
        # _ = plt.plot(times, gp_arr[:,1], "r--", linewidth=2,  markersize=8, label="gp mean $\\Delta \\theta$")

        _ = plt.plot(
            times,
            gp_arr[:, 0],
            "r",
            linewidth=2,
            markersize=8,
            label="gp sample $\\theta$",
        )
        _ = plt.plot(
            times,
            gp_arr[:, 1],
            "r--",
            linewidth=2,
            markersize=8,
            label="gp sample $\\Delta \\theta$",
        )

        igp_arr = np.concatenate(igp_states, axis=0)
        _ = plt.plot(
            times,
            igp_arr[:, 0],
            "b",
            linewidth=2,
            markersize=8,
            label="igp sample $\\theta$",
        )
        _ = plt.plot(
            times,
            igp_arr[:, 1],
            "b--",
            linewidth=2,
            markersize=8,
            label="igp sample $\\Delta \\theta$",
        )

    # plot ground truth
    _ = plt.plot(
        times, test_X[:, 0], "k", linewidth=2, markersize=8, label="true $\\theta$"
    )
    _ = plt.plot(
        times,
        test_X[:, 1],
        "k--",
        linewidth=2,
        markersize=8,
        label="true $\\Delta \\theta$",
    )
    plt.xlabel("Time")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
