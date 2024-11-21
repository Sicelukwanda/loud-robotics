import numpy as np
import matplotlib.pyplot as plt
import pdb
import GPy

from models.IGP import IncrementalGP

def plot_gp(X, m, C, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval 
    plt.fill_between(X[:,0],
                     m[:,0] - 1.96*np.sqrt(np.diag(C)),
                     m[:,0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
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

X = np.array([[0,np.pi/4.0,5*np.pi/7.0,np.pi,3*np.pi/2.0]]).reshape((-1,1))
N = 100
x_test = np.linspace(-3,6,N).reshape(N,1)
y = np.sin(X)


kern = GPy.kern.RBF(input_dim = 1, lengthscale=0.5)
igp = IncrementalGP(X,y, kern, noise_var = 0)


prior_mean = np.zeros_like(x_test).flatten() # (N,) - mean must be one dimensional?
num_samples = 10

f  = np.random.multivariate_normal(mean=prior_mean, cov=igp.kern.K(x_test),size=num_samples)
fig = plt.figure(figsize=(14,6))
for sample in f:
    _ = plt.plot(x_test,sample,'k',linewidth=2)

# plot observations
_ = plt.plot(X,y,'C1+')

fig = plt.figure(figsize=(14,6))

for i in range(N-10):
    xs = x_test[i][:,np.newaxis]
    ys  = igp.predict_xs(xs)

    m, C = igp.predict_X(x_test,full_cov=True)

    plt.clf()
    m_base, C_base = igp.predict(x_test, full_cov=True)
    plot_gp(x_test, m_base, C_base)

    plot_gp(x_test, m, C)
  

    _ = plt.plot(x_test[i],ys,'b+',linewidth=2)

    plt.pause(0.2)

# plot full (unconditioned posterior)

# reset sampling_GP, so we can sample again
igp.reset_sampling_gp()

m, C = igp.predict(x_test, full_cov=True)
plot_gp(x_test, m, C)


fig = plt.figure(figsize=(14,6))
for i in range(N-10):
    xs = x_test[i][:,np.newaxis]
    ys  = igp.predict_xs(xs)

    m, C = igp.predict_X(x_test,full_cov=True)

    plt.clf()
    m_base, C_base = igp.predict(x_test, full_cov=True)
    plot_gp(x_test, m_base, C_base)
    plot_gp(x_test, m, C)
  
    _ = plt.plot(x_test[i],ys,'b+',linewidth=2)

    plt.pause(0.2)

# plot full (unconditioned posterior)
# reset sampling_GP, so we can sample again
igp.reset_sampling_gp()

# m_base, C_base = igp.predict(x_test, full_cov=True)
# plot_gp(x_test, m_base, C_base)

plt.show()