import torch
from torch.distributions import MultivariateNormal
from matplotlib import pyplot as plt

import numpy as np
import GPy

from loud_robotics import InvertedPendulum
from loud_robotics import trajectory_to_transitions

from loud_robotics import IncrementalGPList, GPList
from loud_robotics.models.utils import plot_gp

# plotting options
can_display = False
try:
    import tkinter
    # Test if tkinter actually works
    root = tkinter.Tk()
    root.destroy()
    plt.switch_backend("tkagg")
    can_display = True
except (ImportError, Exception):
    try:
        import PyQt5
        plt.switch_backend("Qt5Agg")
        can_display = True
    except (ImportError, Exception):
        plt.switch_backend('Agg')  # Use non-interactive backend
        can_display = False
        print("Using non-interactive backend for plotting")

plt.rc("font", family="serif", size=14)
# Try to use LaTeX if available, otherwise fall back to regular text
try:
    plt.rc("text", usetex=True)
    plt.rc(
        "text.latex",
        preamble=r"""
           \usepackage{amsmath,amsfonts}
           \renewcommand{\v}[1]{\boldsymbol{#1}}""",
    )
except Exception:
    print("LaTeX not available, using regular text rendering")
    plt.rc("text", usetex=False)

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
                label=f"Particle {i} $\\dot{{\\theta}}$",
                color=env.colors[i],
            )
        plt.legend()

    return trajectories

def apply_pendulum_constraints(gp_list):
    """
    Applies parameter constraints and initializes parameters for a GPList object
    tailored for the inverted pendulum problem.

    Assumes input dimensions are:
    - Angle (theta): index 0
    - Angular velocity (theta_dot): index 1
    - Action (always zero): index 2

    Parameters:
    - gp_list: GPList object containing GP models to which constraints will be applied.
    """

    # Define input dimension indices
    angle_index = 0
    angular_velocity_index = 1
    action_index = 2

    # Loop over each GP model in the GPList
    for gp in gp_list:
        # Ensure the kernel uses ARD (Automatic Relevance Determination)
        # to allow for individual lengthscales per input dimension
        if not hasattr(gp.kern, 'ARD') or not gp.kern.ARD:
            gp.kern = gp.kern.copy()
            gp.kern.ARD = True
            gp.kern.lengthscale = np.ones(gp.input_dim)

        # Constrain lengthscales to be positive
        gp.kern.lengthscale.constrain_positive()

        # Set dimension-specific bounds for lengthscales
        gp.kern.lengthscale[[angle_index]].constrain_bounded(0.01, 10 * np.pi)
        gp.kern.lengthscale[[angular_velocity_index]].constrain_bounded(0.01, 10.0)
        gp.kern.lengthscale[[action_index]].constrain_bounded(1e3, 1e6)

        # Initialize lengthscales
        gp.kern.lengthscale[angle_index] = 1.0
        gp.kern.lengthscale[angular_velocity_index] = 1.0
        gp.kern.lengthscale[action_index] = 1e4  # Large due to lack of variation

        # Constrain kernel variance
        gp.kern.variance.constrain_bounded(1e-6, 1e3)
        # Initialize kernel variance
        gp.kern.variance = 1.0

        # Constrain likelihood variance (noise variance)
        gp.likelihood.variance.constrain_bounded(1e-6, 1e0)
        # Initialize likelihood variance
        gp.likelihood.variance = 1e-5


def main():

    # collect initial data
    # default tensor arguments
    tensor_args = {
        "dtype": torch.float32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # random seed
    seed = 0
    train_seed = 0
    test_seed = 1
    dt = 0.07
    # Pass train_seed and test_seed to the respective environments or simulations


    # environment
    env = InvertedPendulum(
        tensor_args=tensor_args, dt=dt, seed=train_seed
    )  # default dt=0.07

    train_particles = 4
    test_particles = 1
    horizon = 50

    train_trajectories = simulate(
        env, train_particles, horizon, tensor_args, seed, visualize=False, plot=False
    ) 

    # environment
    env = InvertedPendulum(
        tensor_args=tensor_args, dt=dt, seed=test_seed
    )  # default dt=0.07

    # maybe set new start state (for test trajectories)
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

    # save env vizualization
    # env.fig.savefig("pendulum.pdf", bbox_inches='tight', pad_inches=0.1)
    
    # build models

    optimizer_params = {
    'optimizer': 'bfgs', # lbfgs also good
    'messages': True,
    'max_iters': 1000,
    'gtol': 1e-6
    }

    # 1. GP
    # Reset the random seed before igp_list to ensure consistent RNG state
    np.random.seed(0)
    # Create a list of kernels, one for each output dimension
    kernel_list_gp = [
        GPy.kern.RBF(input_dim=train_X.shape[1], ARD=True) for _ in range(train_Y.shape[1])
    ]

    # Initialize the GPList
    noise_variance = 1e-6
    gp_list = GPList(train_X, train_Y, kernel_list_gp, noise_var=noise_variance)

    # apply constraints
    apply_pendulum_constraints(gp_list)

    # Optimize the GP models
    gp_list.optimize(optimizer_params)

    # 2. IGP
    # Reset the random seed before igp_list to ensure consistent RNG state
    np.random.seed(0)

    # Initialize the IncrementalGPList
    kernel_list_igp = [
        GPy.kern.RBF(input_dim=train_X.shape[1], ARD=True) for _ in range(train_Y.shape[1])
    ]

    noise_variance = 1e-6
    igp_list = IncrementalGPList(
        train_X,
        train_Y,
        kernel_list=kernel_list_igp,
        noise_var=noise_variance,
        reoptimize=False,
        reoptim_params_dict=optimizer_params
    )

    # apply constraints
    apply_pendulum_constraints(igp_list)

    # Optimize the IGP models
    igp_list.optimize(optimizer_params)

    # trajectory sampling:

    num_paths = 5
    dummy_action = np.zeros((1, env.action_dim)) 

    # Create subplots: 1 row, 2 columns for Angle and Angular Velocity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Define colors for GP and IGP samples
    gp_color = 'r'
    igp_color = 'b'

    # To manage legends, we'll use labels only once per subplot
    gp_label_plotted = False
    igp_label_plotted = False

    alpha = 0.6 # for samples

    for k in range(num_paths):
        igp_list.reset_sampling_gp()

        igp_states = [test_X[0:1, :-1]]
        gp_states = [test_X[0:1, :-1]]
        times = [0]

        for i in range(1, horizon - 1):
            print(f"sampling step {i}/{horizon} of path {k}/{num_paths}")

            # Incremental GP Prediction and Update
            igp_aug_state = np.concatenate([igp_states[i - 1], dummy_action], axis=1)
            igp_ys = igp_list.predict_xs(igp_aug_state)  # This updates the sampling GPs

            igp_states.append(igp_states[i - 1] + igp_ys)  # prev_state + state_diff

            # GPList Prediction
            gp_aug_state = np.concatenate([gp_states[i - 1], dummy_action], axis=1)
            mu, variance = gp_list.predict(
                gp_aug_state, full_cov=False
            )  # This does NOT update the sampling GPs

            # Sample from predictive posterior
            gp_ys = np.random.multivariate_normal(
                mean=mu.flatten(), cov=np.diag(variance.flatten())
            ).reshape(-1, env.state_dim)

            gp_states.append(gp_states[i - 1] + gp_ys)  # prev_state + state_diff

            times.append(i * env.dt)

        # Convert state lists to arrays
        gp_arr = np.concatenate(gp_states, axis=0)
        igp_arr = np.concatenate(igp_states, axis=0)

        # Plot GP Samples
        if not gp_label_plotted:
            ax1.plot(
                times,
                gp_arr[:, 0],
                "r",
                linewidth=2,
                markersize=8,
                label="Naive GP Sample  $\\theta$",
                alpha=alpha
            )
            ax2.plot(
                times,
                gp_arr[:, 1],
                "r--",
                linewidth=2,
                markersize=8,
                label="Naive GP Sample  $\\dot{\\theta}$",
                alpha=alpha
            )
            gp_label_plotted = True
        else:
            ax1.plot(
                times,
                gp_arr[:, 0],
                "r",
                linewidth=2,
                markersize=8,
                alpha=alpha
            )
            ax2.plot(
                times,
                gp_arr[:, 1],
                "r--",
                linewidth=2,
                markersize=8,
                alpha=alpha
            )

        # Plot IGP Samples
        if not igp_label_plotted:
            ax1.plot(
                times,
                igp_arr[:, 0],
                "b",
                linewidth=2,
                markersize=8,
                label="Recon GP Sample  $\\theta$",
                alpha=alpha
            )
            ax2.plot(
                times,
                igp_arr[:, 1],
                "b--",
                linewidth=2,
                markersize=8,
                label="Recon GP Sample  $\\dot{\\theta}$",
                alpha=alpha
            )
            igp_label_plotted = True
        else:
            ax1.plot(
                times,
                igp_arr[:, 0],
                "b",
                linewidth=2,
                markersize=8,
                alpha=alpha
            )
            ax2.plot(
                times,
                igp_arr[:, 1],
                "b--",
                linewidth=2,
                markersize=8,
                alpha=alpha
            )

    # Plot ground truth on both subplots
    ax1.plot(
        times,
        test_X[:horizon - 1, 0],
        "k",
        linewidth=2,
        markersize=8,
        label="True $\\theta$"
    )
    ax2.plot(
        times,
        test_X[:horizon - 1, 1],
        "k--",
        linewidth=2,
        markersize=8,
        label="True $\\dot{\\theta}$"
    )

    # Set labels and titles
    ax1.set_title("Angle ($\\theta$) Trajectories")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Angle (radians)")

    ax2.set_title("Angular Velocity ($\\dot{\\theta}$) Trajectories")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Angular Velocity (rad/s)")

    # Add legends to both subplots
    ax1.legend()
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots


    # print model parameters
    print("Standard Multi-Output GP")
    for i,gp in enumerate(gp_list):
        print(gp)
        print(f"\nGP Model {i} Parameters:")
        print(f"Lengthscales: {gp.kern.lengthscale.values}")
        print(f"Kernel Variance: {gp.kern.variance.values}")
        print(f"Likelihood Variance: {gp.likelihood.variance.values}")

    print("Incremental Multi-Output GP")
    for i, gp in enumerate(igp_list):
        print(gp)
        print(f"\nIncremental GP Model {i} Parameters:")
        print(f"Lengthscales: {gp.kern.lengthscale.values}")
        print(f"Kernel Variance: {gp.kern.variance.values}")
        print(f"Likelihood Variance: {gp.likelihood.variance.values}")
    
    

    fig.savefig("trajectory_sampling.pdf", bbox_inches='tight', pad_inches=0.1)
    # visualize GP fit

    # x_test
    resolution = 100
    angle_test = np.linspace(-2 * np.pi, 2 * np.pi, resolution).reshape(-1, 1)
    # angle_vel_test = np.linspace(-np.pi, np.pi, resolution).reshape(-1, 1)

    # set all velocities to zero because we are plotting start states
    angle_vel_test = np.zeros((resolution, 1))
    actions_test = np.zeros((resolution, 1))

    X_start_state = np.concatenate([angle_test, angle_vel_test, actions_test], axis=1)

    # mu, var = gp_list.predict(test_X)
    # plot_gp(test_X, mu, var, training_points=(test_X, test_Y))
    mu, var = gp_list.predict(X_start_state)

    fig2 = plt.figure()
    plot_gp(X_start_state, mu, var ) #, training_points=(X_start_state, test_Y))

    # Predict outputs without updating the models
    # mu, cov = igp_list.predict(test_X, full_cov=False)
    mu, cov = igp_list.predict(X_start_state, full_cov=False)
    # mu, cov = igp_list.sampling_gp_predict(X_start_state, full_cov=False)
    

    # Plot the base GP predictions for the all output dimensions
    fig3 = plt.figure()
    # plot_gp(test_X, mu, cov, training_points=(test_X, test_Y))
    plot_gp(X_start_state, mu, cov ) #, training_points=(X_start_state, test_Y))

    # plt.show()
    fig2.savefig("gp.pdf", bbox_inches='tight', pad_inches=0.1)
    fig3.savefig("igp.pdf", bbox_inches='tight', pad_inches=0.1)

    if can_display:
        plt.show()
    else:
        print("Plots saved as trajectory_sampling.pdf, gp.pdf, and igp.pdf")


if __name__ == "__main__":
    main()
