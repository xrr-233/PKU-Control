"""
Classic cart-pole system. 

Study code by Professor Xun Huang, based on OpenAI-Gym. 
2020 July. 

Reference: 
https://gym.openai.com/docs/
"""
import gym
                       
# Test 1
env = gym.make('CartPole-v0')       #load the env
observation = env.reset()           #reset
for t in range(100):        
    env.render()
    print(observation)               # (cart position, Cart Velocity, Pole Angle, Pole Velocity At Tip)
    action = env.action_space.sample()  #(o:push cart to the left; 1: push cart to the right)
    observation, reward, done, info = env.step(action)

env.close()
