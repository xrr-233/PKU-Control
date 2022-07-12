import math

import gym
import numpy as np
from algorithms.DDPG import DDPG

MEMORY_CAPACITY = 10000

def mountain_car_continuous_v0():
    env = gym.make('MountainCarContinuous-v0')
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    RL = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    success=0
    first_episode=0
    for episode in range(100):
        observation = env.reset()
        episode_reward = 0
        for t in range(500):
            env.render()

            # Add exploration noise
            action = RL.choose_action(observation)
            action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration 
            #by using Gaussian distribution with returned action as expectation and var as variance
            observation_, reward, done, info = env.step(action)

            
            prev_x, prev_v = observation
            x, v = observation_
            g = 0.0025
            prev_height = np.sin(3 * prev_x) * .45 + .55  # obtained from documentation
            E1 = prev_height * g + 0.5 * prev_v * prev_v
            height = np.sin(3 * x) * .45 + .55  # obtained form documentation
            E2 = height * g + 0.5 * v * v
            #reward is the same as in mountain_car_v0; the only difference is that we multiply with 100 000 and not 1000
            reward = 100000 * (E2 - E1) 

            RL.store_transition(observation, action, reward / 10, observation_)

            if RL.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                RL.learn()

            observation = observation_
            episode_reward += reward
            if done:
                print('Success', ' Episode:', episode, 'Task: ', t, ' Reward:', episode_reward, 'Variance: %.2f' % var)
                if success==0:
                    first_episode=episode
                success+=1
                break

            elif t==499:
                print('Failure', ' Episode:', episode, 'Task: ', t, ' Reward:', episode_reward, 'Variance: %.2f' % var)

    print('The first episode is: ',first_episode, 'The number of succeses is: ',success)
    env.close()
