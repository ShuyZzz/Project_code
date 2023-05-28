import random
import datetime
import argparse
from pprint import pprint

import gym
import yaml
import torch
import numpy as np
#from munch import munchify

from ddpg_agent import DDPGAgent

env = gym.make('FetchReach-v1')
model= torch.load('save.pt')

state=env.reset()
for t in range(1000):
    env.render()
    observation = state['observation']
    goal = state['desired_goal']
    state_value = torch.from_numpy(np.concatenate((observation, goal)))
    action= model.choose_action_deterministic(state_value)
    next_state,reward,done,info=env.step(action)
    state = next_state

    if done:
        print('done')
        state=env.reset()