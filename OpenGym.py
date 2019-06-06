#!/usr/bin/env python
# coding: utf-8

# ## Load Datasets
# 

# In[1]:


# import sys
# print(sys.version)
# import gym
# import math
# import random
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple
# from itertools import count
# from PIL import Image
# import numpy as np

# import gym
# import numpy as np
# import matplotlib.pyplot as plt

# from JSAnimation.IPython_display import display_animation
# from matplotlib import animation
# from IPython.display import display

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T


from util import * 
from submission import * 



# plt.ion()


# # In[2]:


# from platform import python_version
# print(python_version())
    
# env = gym.make('LunarLander-v2')
# print(env.action_space)
# print(env.observation_space)
# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# # if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


# In[4]:


v_not=3
grid_size=100
thrust=1800
dtheta_range=5
time_penalty=0
fuel_penalty=0
landing_reward=10000000
out_of_bounds_penalty = 1000
args = (v_not,grid_size,thrust,dtheta_range,time_penalty,fuel_penalty,landing_reward,out_of_bounds_penalty)




# In[5]:


mdp = LunarLanderMDP(args)


# In[6]:


ql = QLearningAlgorithm(mdp.actions, mdp.discount(), identityFeatureExtractor,explorationProb=.1)
q_rewards = util.simulate(mdp, ql, 50000)
print("landed ",mdp.landing_count,"times")


# In[8]:


# vi = ValueIteration()
# vi.solve(mdp)
# rl = util.FixedRLAlgorithm(vi.pi)
# vi_rewards = util.simulate(mdp,rl,30000)


# In[64]:


print(float(sum(q_rewards))/len(q_rewards))


# In[ ]:




