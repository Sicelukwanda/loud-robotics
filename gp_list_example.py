import numpy as np
import GPy
import matplotlib.pyplot as plt

from models import GPList
from models.utils import plot_gp

# Generate synthetic data
N = 50
X = np.linspace(0, 10, N).reshape(-1, 1)
Y1 = np.sin(X) + 0.1 * np.random.randn(N, 1)
Y2 = np.cos(X) + 0.1 * np.random.randn(N, 1)
Y = np.hstack((Y1, Y2))  # Shape (N, 2)

# Create a list of kernels, one for each output dimension
kernel_list = [GPy.kern.RBF(input_dim=1) for _ in range(Y.shape[1])] # set  ARD=True if input_dim > 1

optimizer_param_dict = {
'optimizer': 'bfgs', # lbfgs also good
'messages': True,
'max_iters': 1000,
'gtol': 1e-6
}

# Initialize the GPList
gp_list = GPList(X, Y, kernel_list, noise_variance = 1e-6)

# Optimize the GP models
gp_list.optimize(optimizer_param_dict)

# Predict at new input locations
X_new = np.linspace(0, 10, 100).reshape(-1, 1)
mu, var = gp_list.predict(X_new)

# Plot the predictions for the first output dimension
plot_gp(X_new, mu, var, training_points=(X,Y))
plt.show()