import gym
import math
import numpy as np
from algorithms.deep_q_network import DeepQNetwork
def acrobot_v1():
    env = gym.make('Acrobot-v1')
    RL = DeepQNetwork(n_actions=env.action_space.n, n_observations=env.observation_space.shape[0])

    total_step = 0
    success=0
    first_episode=0
    for episode in range(100): 
        observation = env.reset()
        episode_reward = 0 # The final reward score obtained in this round

        for t in range(450):
            env.render()
            action = RL.choose_action(observation) 

            observation_, reward, done, _ = env.step(action) 

            #If the acrobot reached the terminal state, reward him with 10, instead of 0; otherwise, leave -1
            if reward==0:
                reward=10
            
            reward = 1000*reward
            RL.store_transition(observation, action, reward, observation_)
            if(total_step > 100):
                RL.learn()
                

            episode_reward += reward
            observation = observation_

            if done:
                print('Success','Episode:', episode,
                      'Task %d' %t,
                      'Episode_reward %.2f' % episode_reward,
                      'Epsilon %.2f' % RL.epsilon) 
                if success==0:
                    first_episode=episode
                success+=1

                break

            elif t==449:
                print('Failure','Episode:', episode,
                      'Task %d' %t,
                      'Episode_reward %.2f' % episode_reward,
                      'Epsilon %.2f' % RL.epsilon) 

            total_step += 1
    print('The first episode is: ',first_episode, 'The number of succeses is: ',success)

    env.close()