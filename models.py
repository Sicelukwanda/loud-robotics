
import numpy as np

class LinearRegression():
    def __init__(self):
        self.theta_ml = None

    def lr_ml_estimate(self, X, y):
        # X: N x D matrix of training inputs
        # y: N x 1 vector of training targets/observations
        # returns: maximum likelihood parameters (D x 1)
        
        theta_ml = np.linalg.solve(X.T @ X, X.T @ y)
        return theta_ml
    
    def fit(self, X, y):
        """Train the model using the training data."""
        self.theta_ml = self.lr_ml_estimate(X, y)
        return self
    
    def __call__(self, X):
        """Return the predicted values for the input data."""
        if self.theta_ml is None:
            raise ValueError("Model is not trained yet.")
        return X @ self.theta_ml

class GaussianProcess():
    """Gaussian Process Regression model."""
    def __init__(self, kernel, sigma_n=0.01):
        self.kernel = kernel
        self.sigma_n = sigma_n
        self.X = None
        self.y = None
        self.K = None
        self.K_inv = None
        self.theta = None
    
    def fit(self, X, y):
        """Train the model using the training data."""
        self.X = X
        self.y = y
        self.K = self.kernel(X, X) + self.sigma_n**2 * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)
        self.theta = np.linalg.solve(self.K, y)
        return self
    
    def __call__(self, X):
        """Return the predicted values for the input data."""
        if self.X is None:
            raise ValueError("Model is not trained yet.")
        k = self.kernel(X, self.X)
        return k @ self.theta
        

    def long_term_prediction(self, X):
        """Return the long-term prediction for the input data."""
        if self.X is None:
            raise ValueError("Model is not trained yet.")
        k = self.kernel(X, self.X)
        k_star = self.kernel(X, X)
        return k @ self.K_inv @ k_star 