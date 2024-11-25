import GPy
import numpy as np
from .IGP import IncrementalGP


class GPList:
    def __init__(
        self,
        X,
        Y,
        kernel_list=None,
        Y_metadata=None,
        normalizer=None,
        noise_var=1.0,
        mean_function=None,
    ):
        """
        Initializes the GPList with input data X, output data Y, and a list of kernels.

        Parameters:
        - X: Input data of shape (N, M), where N is the number of data points and M is the input dimensionality.
        - Y: Output data of shape (N, D), where D is the number of output dimensions.
        - kernel_list: List of GPy kernel instances, one for each output dimension.
        """
        self.X = X
        self.Y = Y
        self.kernel_list = kernel_list
        self.D = Y.shape[1]  # Number of output dimensions
        self.gp_list = []  # List to store individual GP models

        # Check that the number of kernels matches the number of output dimensions
        if len(kernel_list) != self.D:
            raise ValueError(
                "The length of kernel_list must match the number of output dimensions in Y."
            )

        # Create a GP model for each output dimension
        for d in range(self.D):
            gp = GPy.models.GPRegression(
                X,
                Y[:, d : d + 1],
                kernel_list[d],
                Y_metadata=Y_metadata,
                normalizer=normalizer,
                noise_var=noise_var,
                mean_function=mean_function,
            )
            self.gp_list.append(gp)

    def optimize(self, optim_params_dict={}):
        """
        Optimizes all GP models in the list.

        Parameters:
        - optimizer: Optimization algorithm to use.
        - messages: If True, prints optimization messages.
        - max_iters: Maximum number of iterations for the optimizer.
        """
        for gp in self.gp_list:
            gp.optimize(
                **optim_params_dict
            )

    def predict(self, X_new, full_cov=False):
        """
        Predicts the output at new input locations X_new using all GP models.

        Parameters:
        - X_new: New input data of shape (N_new, M).
        - full_cov: If True, returns the full covariance matrix.

        Returns:
        - mu: Predicted means of shape (N_new, D).
        - var: Predicted variances of shape (N_new, D) if full_cov=False,
               or list of covariance matrices if full_cov=True.
        """
        mu_list = []
        var_list = []
        for gp in self.gp_list:
            mu, var = gp.predict(X_new, full_cov=full_cov)
            mu_list.append(mu)
            var_list.append(var)

        mu = np.hstack(mu_list)  # Concatenate means along the output dimension
        if full_cov:
            # If full covariance is requested, return list of covariance matrices
            return mu, var_list
        else:
            var = np.hstack(var_list)
            return mu, var

    def plot(self, output_dim=0):
        """
        Plots the GP regression for a specified output dimension.

        Parameters:
        - output_dim: Index of the output dimension to plot.
        """
        if output_dim < 0 or output_dim >= self.D:
            raise ValueError("Invalid output dimension index.")
        self.gp_list[output_dim].plot()

    def __getitem__(self, index):
        """
        Allows indexing to access individual GP models.

        Parameters:
        - index: Index of the GP model to access.

        Returns:
        - The GP model at the specified index.
        """
        return self.gp_list[index]

    def __len__(self):
        """
        Returns the number of GP models in the list.
        """
        return len(self.gp_list)


class IncrementalGPList:
    def __init__(
        self,
        X,
        Y,
        kernel_list=None,
        Y_metadata=None,
        normalizer=None,
        noise_var=1.0,
        mean_function=None,
        reoptimize=False,
        reoptim_params_dict={}
    ):
        """
        Initializes the IncrementalGPList with input data X, output data Y, and an optional list of kernels.

        Parameters:
        - X: Input data of shape (N, M), where N is the number of data points and M is the input dimensionality.
        - Y: Output data of shape (N, D), where D is the number of output dimensions.
        - kernel_list: Optional list of GPy kernel instances, one for each output dimension.
                       If None, the same default kernel is used for all output dimensions.
        - reoptimize: If True, re-optimizes the GP model hyperparameters after each update.
        """
        self.X = X
        self.Y = Y
        self.D = Y.shape[1]  # Number of output dimensions
        self.reoptimize = reoptimize
        self.reoptim_params_dict = reoptim_params_dict

        self.gp_list = []  # List to store individual IncrementalGP models

        # If no kernel list is provided, create a default kernel for each output dimension
        if kernel_list is None:
            kernel_list = [GPy.kern.RBF(input_dim=X.shape[1]) for _ in range(self.D)]
        elif len(kernel_list) != self.D:
            raise ValueError(
                "The length of kernel_list must match the number of output dimensions in Y."
            )

        # Create an IncrementalGP model for each output dimension
        for d in range(self.D):
            gp = IncrementalGP(
                X,
                Y[:, d : d + 1],
                kernel=kernel_list[d],
                Y_metadata=Y_metadata,
                normalizer=normalizer,
                noise_var=noise_var,
                mean_function=mean_function,
                reoptimize=reoptimize,
                reoptim_params_dict=self.reoptim_params_dict
            )
            self.gp_list.append(gp)

    def reset_sampling_gp(self):
        """
        Resets the sampling GP for all IncrementalGP models in the list.
        """
        for gp in self.gp_list:
            gp.reset_sampling_gp()

    def _update(self, xs, ys):
        """
        Updates all IncrementalGP models with new data points xs and ys.

        Parameters:
        - xs: New input data of shape (N_new, M).
        - ys: New output data of shape (N_new, D).
        """
        for d, gp in enumerate(self.gp_list):
            gp._update(xs, ys[:, d : d + 1])

    def optimize(self, optim_params_dict={}):
        """
        Optimizes all GP models in the list.

        Parameters:
        - optimizer: Optimization algorithm to use.
        - messages: If True, prints optimization messages.
        - max_iters: Maximum number of iterations for the optimizer.
        """
        for gp in self.gp_list:
            gp.optimize(
                **optim_params_dict
            )

    def predict_xs(
        self,
        xs,
        full_cov=False,
        Y_metadata=None,
        kern_list=None,
        likelihood_list=None,
        include_likelihood=True,
    ):
        """
        Predicts new outputs at locations xs and updates the models incrementally.

        Parameters:
        - xs: New input data of shape (N_new, M).
        - full_cov: If True, returns the full covariance matrix.
        - kern_list: Optional list of kernels to use for prediction.
        - likelihood_list: Optional list of likelihoods to use for prediction.
        - include_likelihood: Whether to include the likelihood variance in the predictions.

        Returns:
        - ys: Predicted outputs of shape (N_new, D).
        """
        ys_list = []
        for d, gp in enumerate(self.gp_list):
            kern = None if kern_list is None else kern_list[d]
            likelihood = None if likelihood_list is None else likelihood_list[d]
            ys = gp.predict_xs(
                xs,
                full_cov=full_cov,
                Y_metadata=Y_metadata,
                kern=kern,
                likelihood=likelihood,
                include_likelihood=include_likelihood,
            )
            ys_list.append(ys)
        ys = np.hstack(ys_list)
        return ys

    def predict_X(
        self,
        xs,
        full_cov=False,
        Y_metadata=None,
        kern_list=None,
        likelihood_list=None,
        include_likelihood=True,
    ):
        """
        Predicts outputs at new input locations xs without updating the models.

        Parameters:
        - xs: New input data of shape (N_new, M).
        - full_cov: If True, returns the full covariance matrices.
        - kern_list: Optional list of kernels to use for prediction.
        - likelihood_list: Optional list of likelihoods to use for prediction.
        - include_likelihood: Whether to include the likelihood variance in the predictions.

        Returns:
        - mu: Predicted means of shape (N_new, D).
        - cov: Predicted variances of shape (N_new, D) if full_cov=False,
               or list of covariance matrices if full_cov=True.
        """
        mu_list = []
        cov_list = []
        for d, gp in enumerate(self.gp_list):
            kern = None if kern_list is None else kern_list[d]
            likelihood = None if likelihood_list is None else likelihood_list[d]
            mu, cov = gp.predict_X(
                xs,
                full_cov=full_cov,
                Y_metadata=Y_metadata,
                kern=kern,
                likelihood=likelihood,
                include_likelihood=include_likelihood,
            )
            mu_list.append(mu)
            cov_list.append(cov)

        mu = np.hstack(mu_list)
        if full_cov:
            # Return list of covariance matrices
            return mu, cov_list
        else:
            cov = np.hstack(cov_list)
            return mu, cov

    def predict(
        self,
        xs,
        full_cov=False,
        Y_metadata=None,
        kern_list=None,
        likelihood_list=None,
        include_likelihood=True,
    ):
        """
        Predicts outputs at new input locations xs using the base GPs.

        Parameters:
        - xs: New input data of shape (N_new, M).
        - full_cov: If True, returns the full covariance matrices.
        - kern_list: Optional list of kernels to use for prediction.
        - likelihood_list: Optional list of likelihoods to use for prediction.
        - include_likelihood: Whether to include the likelihood variance in the predictions.

        Returns:
        - mu: Predicted means of shape (N_new, D).
        - cov: Predicted variances of shape (N_new, D) if full_cov=False,
               or list of covariance matrices if full_cov=True.
        """
        mu_list = []
        cov_list = []
        for d, gp in enumerate(self.gp_list):
            kern = None if kern_list is None else kern_list[d]
            likelihood = None if likelihood_list is None else likelihood_list[d]
            mu, cov = gp.predict(
                xs,
                full_cov=full_cov,
                Y_metadata=Y_metadata,
                kern=kern,
                likelihood=likelihood,
                include_likelihood=include_likelihood,
            )
            mu_list.append(mu)
            cov_list.append(cov)

        mu = np.hstack(mu_list)
        if full_cov:
            # Return list of covariance matrices
            return mu, cov_list
        else:
            cov = np.hstack(cov_list)
            return mu, cov

    def __getitem__(self, index):
        """
        Allows indexing to access individual IncrementalGP models.

        Parameters:
        - index: Index of the IncrementalGP model to access.

        Returns:
        - The IncrementalGP model at the specified index.
        """
        return self.gp_list[index]

    def __len__(self):
        """
        Returns the number of IncrementalGP models in the list.
        """
        return len(self.gp_list)
