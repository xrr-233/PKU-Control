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
env = gym.make('CartPole-v1')

# print states
states = env.observation_space.shape[0]
print('States', states)

# print actions
actions = env.action_space.n
print('Actions', actions)

# Random control
episodes = 10
for episode in range(1,episodes+1):
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
        action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
    print('episode {} score {}'.format(episode, score))


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

# Learning policy
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

policy = EpsGreedyQPolicy()

# Agent and compile and training
sarsa = SARSAAgent(model = model1, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])
sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores = sarsa.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa.save_weights('sarsa_weights.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa.test(env, nb_episodes = 5, visualize= True)
env.close()