import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, cfg, env_params):
        super(Actor, self).__init__()
        self.max_action = env_params.action_max
        self.fc1 = nn.Linear(env_params.obs + env_params.goal, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc3 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.action_out = nn.Linear(cfg.hidden_size, env_params.action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, cfg, env_params):
        super(Critic, self).__init__()
        self.max_action = env_params.action_max
        self.fc1 = nn.Linear(env_params.obs + env_params.goal + env_params.action, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc3 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.q_out = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value