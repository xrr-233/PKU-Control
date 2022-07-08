import gym
from RL_brain.deep_q_network import DeepQNetwork

def cart_pole_v1():
    env = gym.make('CartPole-v1')
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

            x, x_dot, theta, theta_dot = observation_
            # reward1: The more off-center the car is, the reward should be less
            reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # reward2: The more the vertical the bar is, the reward should be higher
            reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = reward1 + reward2
            #The cumulative reward is the sum of these two partial rewards
            RL.store_transition(observation, action, reward, observation_)

            if(total_step > 100):
                RL.learn()

            episode_reward += reward
            observation = observation_

            #goal: the agent can balance the pole for more than 400 tasks
            if (done and t>400) or t==449:
                print('Success','Episode:', episode,
                      'Task %d' %t,
                      'Episode_reward %.2f' % episode_reward,
                      'Epsilon %.2f' % RL.epsilon) 
                if success==0:
                    first_episode=episode

                success+=1
                break

            elif done:
                print('Failure','Episode:', episode,
                      'Task %d' %t,
                      'Episode_reward %.2f' % episode_reward,
                      'Epsilon %.2f' % RL.epsilon)
                break 

            total_step += 1
    print('The first episode is: ',first_episode, 'The number of succeses is: ',success)

    env.close()