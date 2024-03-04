import gymnasium as gym
env = gym.make("CarRacing-v2", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# normal reset, this changes the colour scheme by default
env.reset()

# # reset with colour scheme change
# # env.reset(options={"randomize": True})

# # reset with no colour scheme change
# env.reset(options={"randomize": False})

for i in range(1000):
    env.render()
    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    if terminated:
        print("terminating")
        observation = env.reset()

env.close()