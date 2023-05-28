'''
        # initialize the OU noise process
        ou_noise = np.zeros_like(action)
        ou_noise_mu = 0.0
        ou_noise_theta = 0.15
        ou_noise_sigma = 0.2

        # generate the OU noise
        dx = ou_noise_theta * (ou_noise_mu - ou_noise) + ou_noise_sigma * np.random.randn(*action.shape)
        ou_noise += dx

        # add the OU noise
        action += self.cfg.noise_eps * self.env_params.action_max * ou_noise

        # clip the action within its valid range
        action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)
'''
# add the gaussian noise
        action += self.cfg.noise_eps * self.env_params.action_max * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)

# add the hill-climbing noise
action_grad = np.sign(action) * self.env_params.action_max * 0.01
action += self.cfg.noise_eps * self.env_params.action_max * action_grad
action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)


# add the fine-grained noise
action += self.cfg.noise_eps * self.env_params.action_max * np.random.normal(0, 0.01, action.shape)
action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)


# compute the adaptive noise amplitude
noise_factor = max(0.1, 1.0 - episode_reward / self.cfg.max_episode_reward)
noise_amplitude = self.cfg.noise_eps * self.env_params.action_max * noise_factor

# add the adaptive noise
action += noise_amplitude * np.random.randn(*action.shape)
action = np.clip(action, -self.env_params.action_max, self.env_params.action_max)
