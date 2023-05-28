import gym

env = gym.make('FetchReach-v1')
obs = env.reset()

done = False
while not done:
    # Render the environment (optional)

    # Print the end effector position, goal position, and arm velocity
    end_effector_pos = obs['achieved_goal']
    goal_pos = obs['desired_goal']
    arm_vel = obs['observation'][3:6]  # Extracting arm velocity
    print("End Effector Position:", end_effector_pos)
    print("Goal Position:", goal_pos)
    print("Arm Velocity:", arm_vel)

    action = env.action_space.sample()  # Replace with your own action selection logic
    obs, reward, done, info = env.step(action)

env.close()
