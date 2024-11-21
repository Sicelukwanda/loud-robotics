import numpy as np
import matplotlib.pyplot as plt
import pdb
import GPy
from GPy.util.linalg import tdot

def plot_gp(X, m, C, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval 
    plt.fill_between(X[:,0],
                     m[:,0] - 1.96*np.sqrt(np.diag(C)),
                     m[:,0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])
    
    plt.xlabel("x"), plt.ylabel("f")
    
    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])
np.random.seed(0)

X = np.array([[0,np.pi/4.0,5*np.pi/7.0,np.pi,3*np.pi/2.0]]).reshape((-1,1))
N = 100
x_test = np.linspace(-3,6,N).reshape(N,1)
y = np.sin(X)


kern = GPy.kern.RBF(input_dim = 1, lengthscale=0.5)
gp = GPy.models.GPRegression(X,y, kern, noise_var = 0)


prior_mean = np.zeros_like(x_test).flatten() # (N,) - mean must be one dimensional?
num_samples = 10

f  = np.random.multivariate_normal(mean=prior_mean, cov=gp.kern.K(x_test),size=num_samples)
fig = plt.figure(figsize=(14,6))
for sample in f:
    _ = plt.plot(x_test,sample,'k',linewidth=2)

# plot observations
_ = plt.plot(X,y,'C1+')

fig = plt.figure(figsize=(14,6))
for i in range(N-10):
    xs = x_test[i][:,np.newaxis]
    mu, cov = gp.predict(xs)
    ys = np.random.multivariate_normal(mean=mu.flatten(),cov=cov,size=1)

    m, C = gp.predict(x_test,full_cov=True)

    plt.clf()
    plot_gp(x_test, m, C)
    # update xy
    X_new = np.concatenate([X,xs])
    y_new = np.concatenate([y,ys])
    X = X_new
    y = y_new

    gp.set_XY(X_new,y_new)
    # gp.parameters_changed()
    # gp.optimize()
    _ = plt.plot(x_test[i],ys,'b+',linewidth=2)

    plt.pause(0.2)

m, C = gp.predict(x_test,full_cov=True)

plot_gp(x_test, m, C)
plt.show()