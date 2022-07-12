import gym
import math
import numpy as np
from algorithms.deep_q_network import DeepQNetwork
#Idea for the reward taken from https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
def mountain_car_v0():
    env = gym.make('MountainCar-v0')
    RL = DeepQNetwork(n_actions=env.action_space.n, n_observations=env.observation_space.shape[0])

    total_step = 0
    success=0
    first_episode=0
    for episode in range(100):
        observation = env.reset()
        episode_reward = 0 # The final reward score obtained in this round

        for t in range(190):
            env.render()
            #Let us obtain the position and velocity from the previous state
            prev_x, prev_v=observation
            action = RL.choose_action(observation) 

            observation_, reward, done, _ = env.step(action) 
            x, v=observation_
            #Reward is the increase in mechanical energy, as the car needs to increase its overall energy to be able to climb up
            #the hill
            g=env.gravity
            prev_height=np.sin(3 * prev_x)*.45 + .55 #obtained from documentation  
            E1=prev_height*g + 0.5 * prev_v * prev_v
            height=np.sin(3 * x) * .45 + .55 #obtained form documentation
            E2=height*g+0.5*v*v
            reward = 1000*(E2-E1)
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

            elif t==189:
                print('Failure','Episode:', episode,
                      'Task %d' %t,
                      'Episode_reward %.2f' % episode_reward,
                      'Epsilon %.2f' % RL.epsilon) 
            total_step += 1
    print('The first episode is: ',first_episode, 'The number of succeses is: ',success)
    env.close()