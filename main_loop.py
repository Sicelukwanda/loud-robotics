import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from models import LinearRegression

def unroll_forward(init_state, dynamics_model, action_sequence):
    """Unroll the model from a given initial state for a given sequence of actions."""
    states = [init_state]
    for a in action_sequence:
        s = dynamics_model(np.concatenate([states[-1], a]))
        states.append(s)
    return states

def unfeaturize(feat_x):
    return feat_x #np.array([np.arctan2(feat_x[1],feat_x[0]),feat_x[2]])

def main():
    env = gym.make("Pendulum-v1")
    s = env.reset()
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Starting State:", s)
    s = unfeaturize(s)
    print("Starting State (unfeaturized) shape:", s.shape)
    print("Starting State (unfeaturized):", s)
    
    # Collect traning data
    X = []
    y = []
    for i in range(1000): # 1000 steps = 5 episodes
        a = env.action_space.sample()
        
        sa = np.concatenate([s, a])
        X.append(sa)

        s_, r, done, _ = env.step(a)
        s_ = unfeaturize(s_)
        y.append(s_)  
 
        if done:
            s = unfeaturize(env.reset())
        else:   
            s = s_

    X = np.array(X)
    y = np.array(y)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # evaluate model on training data
    y_predicted = model(X)
    print("Training error:", np.mean((y_predicted - y)**2))

    # plot model predictions vs ground truth
    plt.title("Model predictions vs ground truth (Seen data)")
    plt.plot(y[:,1], label="true")
    plt.plot(y_predicted[:,1], label="predicted")
    plt.legend()
    plt.show()

    # generate random actions
    random_actions = np.random.uniform(
        env.action_space.low, 
        env.action_space.high, 
        size=(env._max_episode_steps,)+env.action_space.shape
        )
    print("random actions shape:",random_actions.shape)

    # unroll model from random initial state
    s = unfeaturize(env.reset())
    states = unroll_forward(s, model, random_actions)
    model_states = np.array(states)

    states = [s]
    for action in random_actions:
        s, _, _, _ = env.step(action)
        s = unfeaturize(s)
        states.append(s)
    env_states = np.array(states)

    # plot model predictions vs ground truth
    plt.title("Model predictions vs ground truth (Unseen data)")
    plt.plot(env_states[:,1], label="true")
    plt.plot(model_states[:,1], label="predicted")
    plt.legend()
    plt.show()

    print("Testing error:", np.mean((model_states - env_states)**2))

if __name__ == "__main__":
    main()