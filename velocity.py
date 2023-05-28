import random
import datetime
import argparse
from pprint import pprint

import gym
import yaml
import torch
import numpy as np
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from ddpg_agent import DDPGAgent


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    pprint(params)
    params = munchify(params)
    return params


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', default='config.yaml',
                        help='The path to the yaml file that contains all configs and hyper-parameters')
    parser.add_argument('--run_name', dest='run_name', default='run')
    parser.add_argument('--use_her', dest='use_her', action='store_true')
    parser.add_argument('--sparse_reward', dest='sparse_reward', action='store_true')
    parser.add_argument('--soft_tau', dest='soft_tau', default=0.001, type=float)
    args = parser.parse_args()
    cfg = load_config(filepath=args.config_path)
    cfg['run_name'] = args.run_name
    cfg['use_her'] = args.use_her
    cfg['sparse_reward'] = args.sparse_reward
    cfg['soft_tau'] = args.soft_tau
    pprint(cfg)
    cfg = munchify(cfg)
    return cfg


def seed_everything(config, environment):
    environment.seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)


if __name__ == '__main__':
    cfg = get_options()
    env = gym.make(cfg.env_name)
    if not cfg.sparse_reward:
        env.env.reward_type = "dense"
    seed_everything(cfg, env)
    env_params = get_env_params(env)

    writer = SummaryWriter('fetchreach_velocity/' + cfg.run_name)

    success_history = []
    episode_reward_history = []
    v_x_history = []
    v_y_history = []
    v_z_history = []

    agent = DDPGAgent(cfg, env_params)

    for episode in range(cfg.n_episodes):
        # Training
        state = env.reset()
        agent.actor.train()
        for t in range(env_params.max_timesteps):
            observation = state['observation']
            goal = state['desired_goal']
            state_value = torch.from_numpy(np.concatenate((observation, goal)))

            action = agent.choose_action(state_value)
            next_state, reward, done, info = env.step(action)

            next_observation = next_state['observation']
            next_goal = next_state['desired_goal']

            next_state_value = np.concatenate((next_observation, next_goal))
            agent.store_experience(state_value, action, reward, next_state_value, done)

            # pretend the goal is what we've already achieved for her sampling
            if cfg.use_her:
                pretend_goal = next_state['achieved_goal']
                pretend_state_value = np.concatenate((observation, pretend_goal))
                next_pretend_state_value = np.concatenate((next_observation, pretend_goal))
                pretend_reward = env.env.compute_reward(pretend_goal, pretend_goal, info)
                agent.store_experience(pretend_state_value, action, pretend_reward, next_pretend_state_value, True)

            agent.learn()
            state = next_state

            # x = observation[0]
            # y = observation[1]
            # z = observation[2]
            #
            # t_x = observation[3]
            # t_y = observation[4]
            # t_z = observation[5]



            v_x = observation[6]
            v_y = observation[7]
            v_z = observation[8]

            position = state['achieved_goal']
            goal = state['desired_goal']

            x = position[0]
            y = position[1]
            z = position[2]

            t_x = goal[0]
            t_y = goal[1]
            t_z = goal[2]

            # dis = observation[9]
            #
            writer.add_scalar('position x', x, t)
            writer.add_scalar('position y', y, t)
            writer.add_scalar('position z', z, t)

            writer.add_scalar('target x', t_x, t)
            writer.add_scalar('target y', t_y, t)
            writer.add_scalar('target z', t_z, t)


            writer.add_scalar('velocity_x', v_x, t)
            writer.add_scalar('velocity_y', v_y, t)
            writer.add_scalar('velocity_z', v_z, t)

            # writer.add_scalar('distance', dis, t)


            if done:
                break




        # Calculate success rate for 100 episodes


        #average_v_x = sum(([0.0] * 100 + v_x_history)[-100:]) / 100
