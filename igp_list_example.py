import numpy as np
import matplotlib.pyplot as plt
import GPy

from models import IncrementalGPList
from models.utils import plot_gp

# Generate synthetic multi-output data
np.random.seed(0)
N = 50
X = np.linspace(0, 10, N).reshape(-1, 1)
Y1 = np.sin(X) + 0.1 * np.random.randn(N, 1)
Y2 = np.cos(X) + 0.1 * np.random.randn(N, 1)
Y = np.hstack((Y1, Y2))  # Shape (N, 2)

# Define kernels for each output dimension
kernel_list = [GPy.kern.RBF(input_dim=1, lengthscale=1.0) for _ in range(Y.shape[1])]

# Initialize the IncrementalGPList
igp_list = IncrementalGPList(
    X, Y, kernel_list=kernel_list, noise_var=0.01, reoptimize=False
)

# Predict at new input locations and update models incrementally
X_new = np.linspace(0, 10, 100).reshape(-1, 1)
num_samples = 5


# Predict outputs without updating the models
mu, cov = igp_list.predict_X(X_new, full_cov=False)

# Plot the base GP predictions for the all output dimensions
#plt.figure(figsize=(10, 6))
# for i in range(Y.shape[1]):
#     plt.plot(X, Y[:, i], "kx", label="Training Data " + str(i))
#     plt.plot(X_new, mu[:, i], label="Predicted Mean " + str(i))
#     plt.fill_between(
#         X_new.flatten(),
#         mu[:, i] - 2 * np.sqrt(cov[:, i]),
#         mu[:, i] + 2 * np.sqrt(cov[:, i]),
#         alpha=0.2,
#         label="Confidence Interval " + str(i),
#     )

# plt.title("IncrementalGP Regression for Output Dimension 1")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()

plot_gp(X_new, mu, cov, training_points=(X,Y))
plt.show()

# Reset sampling GPs (optional)
igp_list.reset_sampling_gp()

for i in range(N - 5):
    xs = X_new[i : i + 1]
    ys = igp_list.predict_xs(xs)  # This updates the models incrementally
    print(f"Sample {i+1}: Input {xs.flatten()}, Predicted Output {ys.flatten()}")
