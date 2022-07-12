import math

import gym
import numpy as np
from algorithms.DDPG import DDPG
#Idea for the reward taken from https://ai-mrkogao.github.io/openai/pendulum/
MEMORY_CAPACITY = 10000

def pendulum_v0():
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    RL = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    dur_task=0
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
            cos_th, sin_th, thdot = observation_ 

            if sin_th>=0:
                th = math.acos(cos_th) 

            else:
                th=2*math.pi-math.acos(cos_th) 
            #th belongs [0,2pi]; in order to use the reward equation provided on the link, we need to normalize it to interval [-pi, pi]
            th=(((th+np.pi) % (2*np.pi)) - np.pi)
            #goal: be slighly off-vertical for more than 100 consecutive tasks
             
            if abs(th)<0.2: #if the pendulum is off-vertical up to  0.2 rad
                dur_task+=1
                if t==499 and dur_task>100:
                    done=True
                    dur_task=0

            else:
                if dur_task>100:
                    done=True
                dur_task=0

            reward = - (th ** 2 + .1 * thdot ** 2 + .001 * (action[0] ** 2))

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
