import numpy as np
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
        reoptimize=False,
    ):

        # flag for if we optimize the model again after adding new data
        self.reoptimize = reoptimize

        super().__init__(X, Y, kernel, Y_metadata, normalizer, noise_var, mean_function)

        self.sampling_gp = None
        self.reset_sampling_gp()

    def reset_sampling_gp(self):
        # resets sampling GP to initial state
        self.sampling_gp = GPy.models.GPRegression.from_gp(self)

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
            self.optimize()

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

        ys = np.random.multivariate_normal(mean=mu.flatten(), cov=cov, size=1)

        self._update(xs, ys)

        return ys

    def predict_X(
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

        return mu, cov
