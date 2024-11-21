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
    import matplotlib.pyplot as plt
    import numpy as np

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
