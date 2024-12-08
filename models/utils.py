from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend("tkagg")
plt.rc("font", family="serif", size=12)
plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"""
       \usepackage{amsmath,amsfonts}
       \renewcommand{\v}[1]{\boldsymbol{#1}}""",
)


def plot_gp(X, m, C, training_points=None, colors=None):
    """
    Plotting utility to plot a GP fit with 95% confidence interval for multiple output dimensions.

    Parameters:
    - X: Input data of shape (N, M)
    - m: Mean predictions of shape (N, D)
    - C: Covariance matrix or variances. If full covariance is provided, it should be a list or array of shape (N, N, D).
         If variances are provided, it should be of shape (N, D).
    - training_points: Tuple (X_train, Y_train), where Y_train is of shape (N_train, D)
    """

    D = m.shape[1]  # Number of output dimensions
    N = X.shape[0]  # Number of data points

    # Check if C is a full covariance matrix or just variances
    if C.ndim == 3:
        # C is a full covariance matrix of shape (N, N, D)
        variances = np.array([np.diag(C[:, :, d]) for d in range(D)]).T  # Shape (N, D)
    else:
        # C is variances of shape (N, D)
        variances = C

    if colors is None:
        colors = [
            "b",
            "g",
            "r",
            "c",
            "m",
            "y",
            "k",
        ]  

    # Check if any figure is currently open
    if not plt.get_fignums():
        plt.figure(figsize=(12, 6))

    for d in range(D):
        color = colors[d % len(colors)]
        plt.plot(X[:, 0], m[:, d], color=color, label=f"Predicted Mean {d}")
        plt.fill_between(
            X[:, 0],
            m[:, d] - 1.96 * np.sqrt(variances[:, d]),
            m[:, d] + 1.96 * np.sqrt(variances[:, d]),
            alpha=0.2,
            color=color,
            label=f"Confidence Interval {d}",
        )

        # Plot training points if included
        if training_points is not None:
            X_train, Y_train = training_points
            plt.plot(
                X_train[:, 0],
                Y_train[:, d],
                "o",
                color=color,
                markersize=5,
                label=f"Training Data {d}",
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GP Fit for Multiple Output Dimensions")
    plt.legend()

def plot_gp_3d(X, Y, m_list, C_list, training_points_list=None, colors_list=None):
    """
    Plotting utility to plot list of GPs in a 3D axes fit with 95% confidence interval for (potentially) multiple output dimensions.
    """

    assert X.shape[1] == 1
    assert Y.shape[1] == 1
    assert len(m_list) == X.shape[0]

    # Create 3D plot
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    B = int(10*X[-1].item())
    H = int(0.3*B)
    W = H
    ax.set_box_aspect((B,W,H))
    breakpoint()
    for i in range(X.shape[0]):
        m = m_list[i]
        C = C_list[i]

        if colors_list is None:
            colors = None
        else:
            colors = colors_list[i]

        D = m.shape[1]  # Number of output dimensions
        N = X.shape[0]  # Number of data points

        # Check if C is a full covariance matrix or just variances
        if C.ndim == 3:
            # C is a full covariance matrix of shape (N, N, D)
            variances = np.array([np.diag(C[:, :, d]) for d in range(D)]).T  # Shape (N, D)
        elif C.ndim == 2 and C.shape[0] == C.shape[1]:
            # C is a full covariance matrix of shape (N, N, D=1)
            variances = np.array([np.diag(C)]).reshape(-1,1)
        else:
            # C is variances of shape (N, D)
            variances = C

        if colors is None:
            colors = [
                "b",
                "g",
                "r",
                "c",
                "m",
                "y",
                "k",
            ]  

        # # Check if any figure is currently open
        # if not plt.get_fignums():
        #     plt.figure(figsize=(12, 6))

        x_vals = np.ones(N)*X.flatten()[i]

        for d in range(D):
            color = colors[d % len(colors)]

            # stack values for surface plot
            
            z_lower = m[:, d] - 1.96 * np.sqrt(variances[:, d])
            z_upper = m[:, d] + 1.96 * np.sqrt(variances[:, d])
            z_vals = np.vstack([z_upper, z_lower])  # Shape: (2, N)
            y_vals  = np.vstack([Y.flatten(), Y.flatten()]) # Shape: (2, N)
            x_vals = np.full_like(y_vals, X.flatten()[i]) # Shape: (2, N)
            
            ax.plot_surface(
                x_vals,
                y_vals,
                z_vals,
                alpha=0.2,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                color=color,
                label=f"Confidence Interval {d}",
            )

            # plot along y_axis as well
            ax.plot(x_vals[0,:], Y, z_lower, color=color)
            ax.plot(x_vals[0,:], Y, z_upper, color=color)
            ax.plot(x_vals[0,:], Y, m[:, d], color=color, label=f"Predicted Mean {d}")

            # Plot training points if included
            # if training_points is not None:
            #     X_train, Y_train = training_points
            #     plt.plot(
            #         X_train[:, 0],
            #         Y_train[:, d],
            #         "o",
            #         color=color,
            #         markersize=5,
            #         label=f"Training Data {d}",
            #     )

    ax.set_xlabel('X-axis', labelpad=15, fontsize=16)
    ax.set_ylabel('Y-axis', labelpad=15, fontsize=16)
    ax.set_zlabel('Z-axis (t)', labelpad=15, fontsize=16)

    # Set limits
    # ax.set_xlim(-5, 100)
    # ax.set_ylim(-2.5, 2.5)
    # ax.set_zlim(-2, 2)

    # Customize tick parameters
    ax.tick_params(axis='x', direction='out', labelsize=12)
    ax.tick_params(axis='y', direction='out', labelsize=12)
    ax.tick_params(axis='z', direction='out', labelsize=12)

    # Add legend
    # ax.legend(loc='upper right', fontsize=12)

    # Add grid
    ax.grid(False)

    # Set background color
    ax.set_facecolor('white')

    # Show the plot
    plt.show()