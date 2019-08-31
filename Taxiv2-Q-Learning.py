# %% Import all libraries
import numpy as np
import gym
import matplotlib.pyplot as plt

# create environment
env = gym.make("Taxi-v2")
env.render()

# %% define hyperparameters
total_episodes = 10000
total_test_episodes = 100
max_steps = 99 # maximum steps each episode

learning_rate = 0.7
gamma = 0.618 # discounting factor

# Exploration parameters
epsilon = 1.0 # exploration rate
max_epsilon = 1.0 # exploration probability at start
min_epsilon = 0.01 # minimum exploration probability 
decay_rate = 0.01 # decay rate for exploration policy

# learn
def q_learning(env, total_episodes=1000, max_steps=99, discount_factor=0.95, learning_rate=0.7, epsilon=1.0, gamma = 0.6, max_epsilon=1.0, min_epsilon=0.05, decay_rate=0.01):
    # 1 create Q table
    action_size = env.action_space.n
    state_size = env.observation_space.n
    Q = np.zeros((state_size, action_size))

    # create stats
    stats= {"steps":{},"reward":{}}

    # 2 iterate over episodes
    for episode in range(total_episodes):
        # 3 Reset the environment
        state = env.reset()
        step = 0
        done = False
        stats["reward"][episode] = 0
        # 4 iterate over steps in episode
        for step in range(max_steps):
            # 5 choose an action
            p = np.random.uniform(0,1)
            if p > epsilon: # exploit
                action = np.argmax(Q[state,:])
            else: # p<epsilon explore 
                action = env.action_space.sample()
            # 6 Act, and receive the reward, observation and info
            new_state, reward, done, info = env.step(action)
            reward += -1
            # log reward
            stats["reward"][episode] += reward
            # 7 update Q
            Q[state, action] = \
                Q[state, action] \
                + learning_rate \
                * (\
                    reward \
                    + gamma * np.max(Q[new_state, :]) \
                    - Q[state, action])
            # 8 update state
            state = new_state
            # 9 if done break
            if done == True: break
        stats["steps"][episode] = step
        # 10 reduce exploration for next episode
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    return Q, stats

Q, stats = q_learning(env)

kr=stats["reward"].keys()
print(list(kr))

vr=stats["reward"].values()
print(list(vr))

ks=stats["steps"].keys()
print(list(ks))

vs=stats["steps"].values()
print(list(vs))



#plt.plot(stats["reward"].keys(), stats["reward"].values())
plt.plot(list(kr), list(vr))
plt.title("total reward per episode")
plt.xlabel("episodes")
plt.ylabel("total reward")
plt.show()

#plt.plot(stats["steps"].keys(), stats["steps"].values())
plt.plot(list(ks), list(vs))
plt.title("steps per episode")
plt.xlabel("episodes")
plt.ylabel("steps")
plt.show()


# %% use
# 1 reset env
state = env.reset()
step = 0
done = False
total_reward = 0
env.render()
# 2 iterate over steps
for step in range(max_steps):
    # 3 choose action
    action = np.argmax(Q[state,:])
    # 4 Act, and receive the reward, observation and info
    state, reward, done, info = env.step(action)
    total_reward += reward
    # 5 render
    print("step",step)
    print("total_reward",total_reward)
    env.render()
    # 6 if done break
    if done == True: break
