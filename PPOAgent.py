import gymnasium as gym

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from rewardFunction import computeLosses


class PPOAgent:
    def __init__(self, env, model=None):
        self.env = env

        if not model:
            self.model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=20,
                gamma=0.99,
            )
        else:
            self.model = PPO.load(model)

    def train(self):
        checkpoint_callback = CheckpointCallback(
            save_freq=2500, save_path="./models/", name_prefix="ppo_car_racing_cont"
        )
        self.model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        self.model.save("ppo_car_racing_final")

def make_env(render_mode="rgb_array", custom_reward=None):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=0.6,
        domain_randomize=True,
        continuous=True,
    )

    original_step = env.step
# 
    def combine_rewards(reg_r, custom_r):
        return (custom_r < 0) * reg_r + (custom_r >= 0) * reg_r * custom_r

    def new_step(action):
        out = original_step(action)
        if len(out) == 4:
            observation, reward, done, info = out
            if custom_reward is not None:
                reward = combine_rewards(reward, 1-custom_reward(action, observation))
            return observation, reward, done, info
        elif len(out) == 5:
            observation, reward, done, info, _ = out
            if custom_reward is not None:
                reward = combine_rewards(reward, 1-custom_reward(action, observation))
            return observation, reward, done, info, _
        else:
            return out

    env.step = new_step
    return env


env = make_vec_env(env_id=lambda: make_env(), n_envs=6)
# env = DummyVecEnv([lambda: make_env(render_mode="human", custom_reward=computeLosses)])
env = VecFrameStack(env, n_stack=6)
agent = PPOAgent(env, model="ppo_car_racing_480000_steps")
observation = env.reset()
agent.train()
# while True:
#     action, _states = agent.model.predict(observation)
#     observation, reward, done, info = env.step(action)
#     print(reward)
#     env.render()

#     if done.any():
#         observation = env.reset()
