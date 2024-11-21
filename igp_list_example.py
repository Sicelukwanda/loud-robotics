import numpy as np
import matplotlib.pyplot as plt
import GPy

from models import IncrementalGPList
from models.utils import plot_gp

# Generate synthetic multi-output data
np.random.seed(0)

X = np.array([[0, np.pi / 4.0, 5 * np.pi / 7.0, np.pi, 3 * np.pi / 2.0]]).reshape(
    (-1, 1)
)
N = X.shape[0]

Y1 = np.sin(X) + 0.1 * np.random.randn(N, 1)
Y2 = np.cos(X) + 0.1 * np.random.randn(N, 1)
Y = np.hstack((Y1, Y2))  # Shape (N, 2)

# Define kernels for each output dimension
kernel_list = [GPy.kern.RBF(input_dim=1, lengthscale=1.0) for _ in range(Y.shape[1])]

# Initialize the IncrementalGPList
noise_variance = 0.0
igp_list = IncrementalGPList(
    X, Y, kernel_list=kernel_list, noise_var=noise_variance, reoptimize=True
)

# Predict at new input locations and update models incrementally
N_test = 100
X_new = np.linspace(0, 10, N_test).reshape(-1, 1)
num_samples = 5


# Predict outputs without updating the models
mu, cov = igp_list.predict_X(X_new, full_cov=False)

# Plot the base GP predictions for the all output dimensions
plot_gp(X_new, mu, cov, training_points=(X, Y))


# Reset sampling GPs (optional)
num_paths = 1
for k in range(num_paths):
    igp_list.reset_sampling_gp()

    for i in range(N_test - 20):

        xs = X_new[i : i + 1]
        ys = igp_list.predict_xs(xs)  # This updates the sampling gps
        print(f"Sample {i+1}: Input {xs.flatten()}, Predicted Output {ys.flatten()}")

        plt.clf()  # get current figure
        m_base, C_base = igp_list.predict_X(X_new, full_cov=False)
        plot_gp(X_new, m_base, C_base, training_points=(X, Y))

        _ = plt.plot(X_new[i, 0], ys, "k+", linewidth=2,  markersize=8)

        plt.pause(0.2)

plt.show()
