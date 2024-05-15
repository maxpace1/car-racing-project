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
            save_freq=10000, save_path="./models/", name_prefix="ppo_car_racing"
        )
        self.model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        self.model.save("ppo_car_racing")

    def predict(self, observation):
        pass


def make_env(render_mode="rgb_array", custom_reward=None):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=1,
        domain_randomize=False,
        continuous=False,
    )

    original_step = env.step

    ##############################
    # 1.0 is full custom reward, 0.0 is full default reward (from environment)
    CUSTOM_REWARD_WEIGHT = 0.5
    ##############################

    def new_step(action):
        out = original_step(action)
        if len(out) == 4:
            observation, reward, done, info = out
            if custom_reward is not None:
                reward = (
                    CUSTOM_REWARD_WEIGHT * custom_reward(action, observation)
                    + (1 - CUSTOM_REWARD_WEIGHT) * reward
                )
            return observation, reward, done, info
        elif len(out) == 5:
            observation, reward, done, info, _ = out
            if custom_reward is not None:
                reward = (
                    CUSTOM_REWARD_WEIGHT * custom_reward(action, observation)
                    + (1 - CUSTOM_REWARD_WEIGHT) * reward
                )
            return observation, reward, done, info, _
        else:
            return out

    env.step = new_step
    return env


env = make_vec_env(env_id=make_env, n_envs=4, custom_reward=computeLosses)
# env = DummyVecEnv([lambda: make_env(render_mode="human", custom_reward=computeLosses)])
env = VecFrameStack(env, n_stack=6)
agent = PPOAgent(env, model="ppo_car_racing_480000_steps")
observation = env.reset()
agent.train()
while True:
    action, _states = agent.model.predict(observation)
    observation, reward, done, info = env.step(action)
    print(reward)
    env.render()

    if done.any():
        observation = env.reset()
