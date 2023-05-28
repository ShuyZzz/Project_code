import gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer

env = gym.make("FetchReach-v1")
log_dir = './bs3_tensorboard/'

# DDPG
model = DDPG(policy="MultiInputPolicy", env=env,
    buffer_size=500000,learning_rate=0.001, batch_size=64, tau=0.001,
             replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=200000)


'''
# TD3
model = TD3(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=20000)
model.save("td3_panda_reach_v2")
# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=20000)
model.save("sac_panda_reach_v2")
'''