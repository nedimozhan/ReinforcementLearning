# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 01:37:07 2022

@author: nedim
"""



# env=gym.make("Taxi-v3").env

# q_table = np.zeros([env.observation_space.n,env.action_space.n])

# alpha = 0.1
# gamma = 0.9
# epsilon = 0.1

# reward_list = []
# droputs_list = []


# episode_number = 1000

# for i in range(1, episode_number):
    
#     state = env.reset()
#     reward_count = 0
#     droput_count = 0
    
    
#     while True:
        
#         if random.uniform(0, 1) < epsilon:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(q_table[state])
        
#         next_state,reward,done, _ = env.step(action)
        
#         # Q-Learning
#         old_value = q_table[state][action]
#         next_max = np.max(q_table[state])
#         next_value = (1-alpha)*old_value + alpha*(reward+gamma*next_max)
#         q_table[state][action] = next_value
#         state=next_state
        
#         if(reward == -10):
#             droput_count += 1
        
#         if done:
#             break
        
#     if (i%10 == 0):
#         reward_count +=  reward
#         droputs_list.append(droput_count)
#         reward_list.append(reward_count)
#         print("Episode : {},Reward : {}, Wrong Dropout : {}".format(i,reward_count,droput_count))


#%%
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1").env
Q_table=np.zeros([env.observation_space.n,env.action_space.n])
env.render()
from gym.envs.registration import register
register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name":"4x4","is_slippery" : False},
    max_episode_steps=100,
    reward_threshold=0.78,
)

exploreCoefficient=0.1
learningRate=0.9
careFuture=0.9

time=1000000

for i in range(1, time):
    reward_count=0
    punish_count=0
    state = env.reset()
    
    while True:
        
         if (random.uniform(0, 1)>exploreCoefficient):
             action = env.action_space.sample()
         else:
             action = np.argmax(Q_table[state])
        
         next_step,reward,done,_ = env.step(action)
        
         old_value=Q_table[state][action]
         next_max=np.max(Q_table[next_step])
         next_value = (1-learningRate)*old_value + learningRate*(reward+careFuture*next_max)
         Q_table[state][action]=next_value
         state=next_step
        
        
         if reward==1:
             reward_count+=1
         else:
             punish_count+=1
         
         if done:
             break
    if(i%100==0):  
        print("Reward Count : {},Punish Count : {}".format(reward_count, punish_count))
    
    
    
