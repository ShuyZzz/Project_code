import torch
import torch.optim as optim

from utils import *
from models import Actor, Critic
from memory import ExperienceReplayMemory


class DDPGAgent(object):
    def __init__(self, cfg, env_params):
        self.cfg = cfg
        self.env_params = env_params
        self.memory_size = cfg.buffer_size

        self.actor = Actor(cfg, env_params).cpu()
        self.critic = Critic(cfg, env_params).cpu()
        self.actor_target = Actor(cfg, env_params).cpu()
        self.critic_target = Critic(cfg, env_params).cpu()

        # make sure source and target network has the same parameters
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.actor_target_optim = optim.RMSprop(self.actor_target.parameters(), lr=self.cfg.lr_actor)
        self.critic_target_optim = optim.RMSprop(self.critic_target.parameters(), lr=self.cfg.lr_critic)

        self.replay_memory = ExperienceReplayMemory(cfg, env_params)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_memory.add_experience(state=state, action=action,
                                          reward=reward, next_state=next_state,
                                          done=done)

    def get_sample_experience(self):
        state, action, reward, next_state, done = self.replay_memory.get_random_experience(self.cfg.batch_size)

        t_state = torch.tensor(state).cpu()
        t_action = torch.tensor(action).cpu()
        t_reward = torch.tensor(reward).cpu()
        t_next_state = torch.tensor(next_state).cpu()
        t_done = torch.tensor(done).cpu()

        return t_state, t_action, t_reward, t_next_state, t_done

    def choose_action(self, observation):
        pi = self.actor(observation.cpu().float())
        action = pi.detach().cpu().numpy().squeeze()

        # add the gaussian noise
        action += self.cfg.noise_eps * self.env_params.action_max * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)



        # random actions...
        random_actions = np.random.uniform(low=-self.env_params.action_max, high=self.env_params.action_max,
                                           size=self.env_params.action)
        # choose if use the random actions
        action += np.random.binomial(1, self.cfg.random_eps, 1)[0] * (random_actions - action)
        return action

    def choose_action_deterministic(self, observation):
        # choose action without noise and without epsilon randomness
        with torch.no_grad():
            pi = self.actor(observation.cpu().float())
        action = pi.detach().cpu().numpy().squeeze()
        return action

    def learn(self):
        if self.replay_memory.counter < self.cfg.batch_size:
            return

        state, action, reward, next_state, done = self.get_sample_experience()
        # Gets the evenly spaced batches

        q_value = self.critic(state, action)

        next_actions = self.actor_target(next_state)
        next_q_value = self.critic_target(next_state, next_actions.detach())

        if self.cfg.sparse_reward:
            q_prime = reward[:, None] + self.cfg.gamma * done[:, None] * next_q_value
        else:
            q_prime = reward[:, None] + self.cfg.gamma * next_q_value

        critic_loss = torch.nn.functional.mse_loss(q_value, q_prime)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update target network from source network with a certain ratio
        soft_update(self.actor_target, self.actor, self.cfg.soft_tau)
        soft_update(self.critic_target, self.critic, self.cfg.soft_tau)