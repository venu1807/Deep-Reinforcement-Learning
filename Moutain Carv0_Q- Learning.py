import sys
import gym
from gym import wrappers, logger
from pprint import pprint
import numpy as np


class QLearningAgent():
    def __init__(self, action_space, observation_space):
        self.action_space = action_space.n  # self.action_space will be a number n. You can return any integer x with 0 <= x <= n in act 
        self.observation_space = observation_space
        self.buckets_per_dimension = 40  # Defines how many discrete bisn are used in each dimesnion of the observation.
        self.q_table = np.zeros((self.buckets_per_dimension ** 2, self.action_space))
        self.last_state = None
        self.last_action = None

    def act(self, observation, last_reward, episode):
        # Return 0 = push left, 1 = no push, 2 = push right
        learning_rate = max(0.001, 1.0 * (0.85 ** int(episode/100)))
        lookahead = 1        
        state = self.to_state(observation)
        if self.last_state is not None:
            self.q_table[self.last_state][self.last_action] = self.q_table[self.last_state][self.last_action] \
                + learning_rate * (last_reward + lookahead * np.max(self.q_table[state]) - self.q_table[self.last_state][self.last_action])
        possible_actions = self.q_table[state]
        probabilities = np.exp(possible_actions) / np.sum(np.exp(possible_actions))
        choice = np.random.choice(self.action_space, p=probabilities)
        self.last_state = state
        self.last_action = choice
        return choice


    def to_state(self, observation):
        upper_bound_position = self.observation_space.high[0]
        lower_bound_position = self.observation_space.low[0]
        upper_bound_velocity = self.observation_space.high[1]
        lower_bound_velocity = self.observation_space.low[1]
        step_size_position = (upper_bound_position - lower_bound_position) / self.buckets_per_dimension
        step_size_velocity = (upper_bound_velocity - lower_bound_velocity) / self.buckets_per_dimension
        bucket_position = int((observation[0] - lower_bound_position) / step_size_position)
        bucket_velocity = int((observation[1] - lower_bound_velocity) / step_size_velocity)
        return bucket_position * self.buckets_per_dimension + bucket_velocity

    
    def get_best_actions(self):
        return np.argmax(self.q_table, axis=1)

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make("MountainCar-v0").env

    env.seed(0)
    agent = QLearningAgent(env.action_space, env.observation_space)

    episode_count = 1000
    attempts_in_episode = 10000

    for ep in range(episode_count):
        ob = env.reset()
        reward = 0
        action = None
        done = False
        for j in range(attempts_in_episode):
            action = agent.act(ob, reward, ep)
            ob, reward, done, _ = env.step(action)
            # env.render()
            if done:
                print("Flag reached in episode:{}".format(ep))
                break
    
    input("Press Enter to continue and show best solution...")
    ob = env.reset()    
    best = agent.get_best_actions()
    done = False
    while not done:
        action = best[agent.to_state(ob)]
        ob, _, done, _ = env.step(action)
        env.render()
    print("Done") 

    env.close()

