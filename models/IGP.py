import numpy as np
from numpy.linalg import LinAlgError, cholesky
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
import GPy
from GPy.util.linalg import tdot


class IncrementalGP(GPy.models.GPRegression):
    """
    This is a wrapper class for GPy.models.GPRegression.
    The incremental GP, conditions the GP on the latest sample, i.e., adds it to the data
    """

    def __init__(
        self,
        X,
        Y,
        kernel=None,
        Y_metadata=None,
        normalizer=None,
        noise_var=1.0,
        mean_function=None,
        jitter = 1e-5,
        reoptimize=False,
        reoptim_params_dict = {}
    ):

        # flag for if we optimize the model again after adding new data
        self.reoptimize = reoptimize
        self.reoptim_params_dict = reoptim_params_dict

        super().__init__(X, Y, kernel, Y_metadata, normalizer, noise_var, mean_function)

        self.sampling_gp = None

        # save a copy of the initial GP model so we can "reset"
        self.initial_gp_model = GPy.models.GPRegression.from_gp(self) # this uses deepcopy
        self.reset_sampling_gp()

        self.jitter = jitter # default 1e-5

    def reset_sampling_gp(self):
        # resets sampling GP to initial "state", i.e., before conditioning on samples
        self.sampling_gp = GPy.models.GPRegression.from_gp(self.initial_gp_model)

    def optimize(self, **kwargs):
        """Wrap optimize so that we update the initial_gp_model"""
        result = super().optimize(**kwargs)
        # update the initial GP
        self.initial_gp_model = GPy.models.GPRegression.from_gp(self)

        # reset sampling gp
        self.reset_sampling_gp()
        return result


    def _update(self, xs, ys):
        """appends latest sample(s): xs (N* x D), ys (N*,) to the training data X,y
        and recomputes the kernel matrix and other quantities"""

        # update the GP model
        self.sampling_gp.update_model(False)
        self.sampling_gp.set_XY(
            np.concatenate([self.sampling_gp.X, xs]),
            np.concatenate([self.sampling_gp.Y, ys]),
        )
        self.sampling_gp.update_model(True)

        # reoptimize the model
        if self.reoptimize:
            self.sampling_gp.optimize(**self.reoptim_params_dict)

    def predict_xs(
        self,
        xs,
        full_cov=False,
        Y_metadata=None,
        kern=None,
        likelihood=None,
        include_likelihood=True,
    ):

        mu, cov = self.sampling_gp.predict(
            xs, full_cov, Y_metadata, kern, likelihood, include_likelihood
        )

        # mitigate numerical issues 
        try:
            L = cholesky(cov)
        except LinAlgError as e:
            print(str(e))
            # add jitter and try again
            cov += np.eye(cov.shape[0]) * jitter 
            L = cholesky(cov)
        # ys = np.linalg.cholesky(L) @ np.random.normal(size=mu.shape)
        # ys = np.random.multivariate_normal(mean=mu.flatten(), cov=cov).reshape(-1,1)
        ys = mu.flatten() + L @ np.random.randn(cov.shape[0])
        ys = ys.reshape(-1, 1)

        # TODO: only update the model if xs is not already present in the data.

        # Initialize lists to collect new data points
        new_xs = []
        new_ys = []

        # Loop over each input-output pair
        for x_i, y_i in zip(xs, ys):
            # Check if x_i is close to any row in the existing data
            if not np.any(np.all(np.isclose(self.sampling_gp.X, x_i, atol=self.jitter), axis=1)):
                new_xs.append(x_i)
                new_ys.append(y_i)
            else:
                # Optional: print a message or handle duplicates as needed
                print(f"Input {x_i} is already in the training data. Skipping update.")

        # Update the model with new data points if there are any
        if new_xs:
            new_xs = np.vstack(new_xs)
            new_ys = np.vstack(new_ys)
            self._update(new_xs, new_ys)

        return ys

    def sampling_gp_predict(
        self,
        xs,
        full_cov=False,
        Y_metadata=None,
        kern=None,
        likelihood=None,
        include_likelihood=True,
    ):
        """
        Make a prediction with the sampling GP
        """

        mu, cov = self.sampling_gp.predict(
            xs, full_cov, Y_metadata, kern, likelihood, include_likelihood
        )

        return mu, cov
