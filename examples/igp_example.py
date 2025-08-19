import numpy as np
import matplotlib.pyplot as plt
import GPy

from loud_robotics.models import IncrementalGP

# Check if we can use interactive backend
try:
    import tkinter
    can_display = True
except ImportError:
    try:
        import PyQt5
        can_display = True
    except ImportError:
        can_display = False
        plt.switch_backend('Agg')  # Use non-interactive backend
        print("No interactive backend available. Saving static images instead.")

def plot_gp(X, m, C, training_points=None):
    """Plotting utility to plot a GP fit with 95% confidence interval"""
    # Plot 95% confidence interval
    plt.fill_between(
        X[:, 0],
        m[:, 0] - 1.96 * np.sqrt(np.diag(C)),
        m[:, 0] + 1.96 * np.sqrt(np.diag(C)),
        alpha=0.5,
    )
    # Plot GP mean and initial training points
    plt.plot(X, m, "-", alpha=0.8)
    plt.legend(labels=["GP fit"])

    plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])


np.random.seed(0)

X = np.array([[0, np.pi / 4.0, 5 * np.pi / 7.0, np.pi, 3 * np.pi / 2.0]]).reshape(
    (-1, 1)
)
N = 100
x_test = np.linspace(-3, 6, N).reshape(N, 1)
y = np.sin(X)


kern = GPy.kern.RBF(input_dim=1, lengthscale=0.5) # set ARD=True if multi-dim inputs
igp = IncrementalGP(X, y, kern, noise_var=0)


prior_mean = np.zeros_like(x_test).flatten()  # (N,) - mean must be one dimensional?
num_samples = 10

f = np.random.multivariate_normal(
    mean=prior_mean, cov=igp.kern.K(x_test), size=num_samples
)
fig = plt.figure(figsize=(14, 6))
for sample in f:
    _ = plt.plot(x_test, sample, "k", linewidth=2)

# plot observations
_ = plt.plot(X, y, "C1+")
plt.title("Prior Samples from GP")
plt.grid(True, alpha=0.3)

if can_display:
    plt.show()  # Show the initial prior samples
    
    # Turn on interactive mode for the incremental updates
    plt.ion()
    
    # Create a single figure for the incremental updates
    fig = plt.figure(figsize=(14, 6))
    
    for i in range(N - 90):
        xs = x_test[i][:, np.newaxis]
        ys = igp.predict_xs(xs)

        m, C = igp.sampling_gp_predict(x_test, full_cov=True)

        plt.clf()  # Clear the figure
        m_base, C_base = igp.predict(x_test, full_cov=True)
        plot_gp(x_test, m_base, C_base)
        plot_gp(x_test, m, C)
        _ = plt.plot(x_test[i], ys, "b+", linewidth=2, markersize=8)
        plt.title(f"Incremental GP - Step {i+1}/{N-90}")
        plt.grid(True, alpha=0.3)
        
        plt.pause(0.1)  # Shorter pause for smoother animation
        plt.draw()

    plt.ioff()  # Turn off interactive mode
    
    # Show final result
    igp.reset_sampling_gp()
    m, C = igp.predict(x_test, full_cov=True)
    plt.figure(figsize=(14, 6))
    plot_gp(x_test, m, C)
    plt.title("Full Unconditioned Posterior")
    plt.grid(True, alpha=0.3)
    plt.show()
    
else:
    # Non-interactive mode: save key steps
    plt.savefig("igp_prior.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Creating static snapshots of incremental GP process...")
    steps_to_save = [0, (N-90)//4, (N-90)//2, 3*(N-90)//4, N-91]
    
    for step_idx, i in enumerate(steps_to_save):
        if i >= N - 90:
            continue
            
        xs = x_test[i][:, np.newaxis]
        ys = igp.predict_xs(xs)

        m, C = igp.sampling_gp_predict(x_test, full_cov=True)

        plt.figure(figsize=(14, 6))
        m_base, C_base = igp.predict(x_test, full_cov=True)
        plot_gp(x_test, m_base, C_base)
        plot_gp(x_test, m, C)
        _ = plt.plot(x_test[i], ys, "b+", linewidth=2, markersize=8)
        plt.title(f"Incremental GP - Step {i+1}/{N-90}")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"igp_step_{step_idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(steps_to_save)} snapshots: igp_step_1.png to igp_step_{len(steps_to_save)}.png")

    # plot full (unconditioned posterior)
    igp.reset_sampling_gp()
    m, C = igp.predict(x_test, full_cov=True)
    plt.figure(figsize=(14, 6))
    plot_gp(x_test, m, C)
    plt.title("Full Unconditioned Posterior")
    plt.grid(True, alpha=0.3)
    plt.savefig("igp_final.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved final plot: igp_final.png")
