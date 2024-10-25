from time import time
import mujoco
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
# import mujoco_viewer
import GPy
import time
from sklearn.model_selection import train_test_split


# load saved data
states = np.load("states.npy")
augmented_states = np.load("aug_states.npy")
time_hist = np.load("time_hist.npy")

# Train-test split
train_augmented_states, test_augmented_states, train_states, test_states = train_test_split(
    augmented_states[:-1,:], 
    states[1:,:], 
    test_size=0.3, 
    random_state=42,
    shuffle=False
    )

# define the GP model
# TODO: use non-iso RBF kernel
kernel_list = [GPy.kern.RBF(augmented_states.shape[1], variance=1.0, lengthscale=1.0, ARD=True) for _ in range(12)]
# TODO: Check if these are separate independent kernels or if they are combined
kernel = GPy.kern.Add(kernel_list)
print(kernel)

# # fit multiple-output GP model (one kernel for each pressure channel)
# model = GPy.models.GPRegression(train_augmented_states, train_states, kernel)
# model.optimize()
# print(model)

# fit a GP model for each pressure channel
gp_list = []
start_time = time.time()
for i in range(12):
    gp = GPy.models.GPRegression(train_augmented_states, np.expand_dims(train_states[:, i], axis=1), kernel_list[i])
    gp.optimize()
    gp.likelihood.variance = max(gp.likelihood.variance, 1e-6)
    gp_list.append(gp)

    print(f"Time to fit GP {i}th model:", time.time() - start_time)

    print(gp_list[-1])

# make train predictions
start_time = time.time()
train_pred_list = []
train_pred_list_var = []
for i in range(12):
    pred = gp_list[i].predict(train_augmented_states)
    train_pred_list.append(pred[0]) # mean
    train_pred_list_var.append(pred[1]) # variance

print("Time to make train predictions:", time.time() - start_time)
train_pred_states = np.concatenate(train_pred_list, axis=1)
train_pred_states_var = np.concatenate(train_pred_list_var, axis=1)
print("min variance:", np.min(train_pred_states_var), "max variance:", np.max(train_pred_states_var))

train_std_lower = train_pred_states - 2 * np.sqrt(train_pred_states_var)
train_std_upper = train_pred_states + 2 * np.sqrt(train_pred_states_var)

# make train predictions
start_time = time.time()
test_pred_list = []
test_pred_list_var = []
for i in range(12):
    pred = gp_list[i].predict(test_augmented_states)
    test_pred_list.append(pred[0]) # mean
    test_pred_list_var.append(pred[1]) # variance

print("Time to make test predictions:", time.time() - start_time)
test_pred_states = np.concatenate(test_pred_list, axis=1)
test_pred_states_var = np.concatenate(test_pred_list_var, axis=1)
test_std_lower = test_pred_states - 2 * np.sqrt(test_pred_states_var)
test_std_upper = test_pred_states + 2 * np.sqrt(test_pred_states_var)

train_times = np.linspace(0,time_hist[-1],train_states.shape[0])
test_times = np.linspace(0,time_hist[-1],test_states.shape[0])

# plot the predictions vs the actual states
plt.figure("Predictions vs actual positions")
for i in range(3): 
    plt.subplot(3, 1, i+1)
    # plt.plot(train_times, train_states[:, 2*i], label=f"actual u{i}")
    # plt.plot(train_times, train_pred_states[:, 2*i], label=f"train_pred u{i}")
    # # plot train prediction std dev
    # plt.fill_between(train_times, train_std_lower[:, 2*i], train_std_upper[:, 2*i], alpha=0.2)

    plt.plot(test_times, test_states[:, 2*i], label=f"actual test u{i}")
    plt.plot(test_times, test_pred_states[:, 2*i], label=f"pred test u{i}")
    # plot test prediction std dev
    plt.fill_between(test_times, test_std_lower[:, 2*i], test_std_upper[:, 2*i], alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position (rad)")
    plt.legend()

plt.figure("Predictions vs actual velocities")
for i in range(3):
    plt.subplot(3, 1, i+1)
    # plt.plot(train_times, train_states[:, 2*i+1], label=f"actual v{i}")
    # plt.plot(train_times, train_pred_states[:, 2*i+1], label=f"train_pred v{i}")
    # # plot train prediction std dev
    # plt.fill_between(train_times, train_std_lower[:, 2*i+1], train_std_upper[:, 2*i+1], alpha=0.2)

    plt.plot(test_times, test_states[:, 2*i+1], label=f"actual test v{i}")
    plt.plot(test_times, test_pred_states[:, 2*i+1], label=f"pred test v{i}")
    # plot test prediction std dev
    plt.fill_between(test_times, test_std_lower[:, 2*i+1], test_std_upper[:, 2*i+1], alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
plt.show()