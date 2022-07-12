"""
Classic cart-pole system.
The study code is prepared by Professor Xun Huang for the teaching purpose.
The code is based on OpenAI-Gym.
2020 July.
Reference:
https://gym.openai.com/docs/
"""
# Test
import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
#from keras.models import Sequential
from keras.models import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory

from deep_q_network import DeepQNetwork

# Learning policy
from rl.agents import SARSAAgent, DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy

env = gym.make('CartPole-v1')
np.random.seed(123)
env.seed(123)

# print states
states = env.observation_space.shape[0]
print('States', states)
# print actions
actions = env.action_space.n
print('Actions', actions)

'''这只是一个示例，实际上没有啥用'''
'''
RL = DeepQNetwork(n_actions=env.action_space.n, n_observations=env.observation_space.shape[0])

# DQN control
episodes = 10
total_step = 0
for episode in range(1, episodes + 1):
    # At each begining reset the game 
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:  # When done= True, the game is lost  
        # visualize each step
        env.render()
        # choose a random action
        action = RL.choose_action(state)
        # execute the action
        n_state, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = n_state
        # reward1: The more off-center the car is, the reward should be less
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # reward2: The more the vertical the bar is, the reward should be higher
        reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = reward1 + reward2
        # The cumulative reward is the sum of these two partial rewards
        RL.store_transition(state, action, reward, n_state)
        if (total_step > 100):
            RL.learn()
        # keep track of rewards
        score += reward
        state = n_state
        total_step += 1
    print('episode {} score {}'.format(episode, score))
'''
# Define a smart agent (a very small network)
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
model1 = agent(env.observation_space.shape[0], env.action_space.n)
model1.summary()

'''这是原代码，使用保守的SARSA算法，效果不好，仅用于测试网络用'''
'''
policy = EpsGreedyQPolicy()

# Agent and compile and training
sarsa = SARSAAgent(model = model1, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])
sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
# Then, test the trained model
#scores = sarsa.test(env, nb_episodes = 100, visualize=True)
env.close()
# Next, save the model
sarsa.save_weights('sarsa_weights.h5f', overwrite=True)

env = gym.make('CartPole-v1')

# Define a smart agent (a very small network)
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model1 = agent(env.observation_space.shape[0], env.action_space.n)
model1.summary()

policy = EpsGreedyQPolicy()

# Agent and compile and training
sarsa = SARSAAgent(model = model1, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])

# load the weights
sarsa.load_weights('sarsa_weights.h5f')
_ = sarsa.test(env, nb_episodes = 5, visualize= True)
env.close()
'''

'''现在更改成DQN，效果好，作为最终方案'''
'''要重新训练的话删注释'''
'''
policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

# Agent and compile and training
dqn = DQNAgent(model=model1, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
# Then, test the trained model
#scores = dqn.test(env, nb_episodes = 100, visualize=True)
env.close()
# Next, save the model
dqn.save_weights('dqn_weights.h5f', overwrite=True)
'''

env = gym.make('CartPole-v1')

# Define a smart agent (a very small network)
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model1 = agent(env.observation_space.shape[0], env.action_space.n)
model1.summary()

policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

# Agent and compile and training
dqn = DQNAgent(model=model1, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# load the weights
dqn.load_weights('dqn_weights.h5f')
_ = dqn.test(env, nb_episodes = 5, visualize= True)
env.close()