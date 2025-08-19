
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

