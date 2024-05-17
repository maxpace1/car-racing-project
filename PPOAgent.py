import gymnasium as gym

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from rewardFunction import computeLosses

import matplotlib.pyplot as plt
import time


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
            save_freq=2500,
            save_path="./models/",
            name_prefix="ppo_car_racing_cont_600000_steps",
        )
        self.model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        self.model.save("ppo_car_racing_final")


def make_env(render_mode="rgb_array", custom_reward=None):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=1,
        domain_randomize=True,
        continuous=False,
    )

    original_step = env.step

    env.gas = 0
    env.turn = 0
    env.brake = 0

    def combine_rewards(reg_r, custom_r):
        return (custom_r < 0) * reg_r + (custom_r >= 0) * reg_r * custom_r

    def new_step(action):
        out = original_step(action)
        if len(out) == 4:
            observation, reward, done, info = out
        elif len(out) == 5:
            observation, reward, done, info, _ = out
        if custom_reward is not None:

            match action:
                case 1:
                    env.turn -= 0.2
                case 2:
                    env.turn += 0.2
                case 3:
                    env.gas += 0.2
                case 4:
                    env.gas -= 0.2
                    env.brake += 0.2

            if env.turn > 0:
                env.turn -= 0.1
                env.turn = min(env.turn, 1.0)
            if env.turn < 0:
                env.turn += 0.1
                env.turn = max(env.turn, -1.0)
            env.gas -= 0.1
            env.brake -= 0.1
            env.gas = min(max(env.gas, 0.0), 1.0)
            env.brake = min(max(env.brake, 0.0), 1.0)

            reward = combine_rewards(
                reward,
                1
                - custom_reward(
                    action,
                    observation,
                    converted=(env.turn, env.gas, env.brake),
                ),
            )

            return observation, reward, done, info, _
        else:
            return out

    env.step = new_step
    return env


############## MAIN ##############

# env = make_vec_env(env_id=lambda: make_env(), n_envs=6)
env = DummyVecEnv([lambda: make_env(render_mode="human", custom_reward=computeLosses)])
env = VecFrameStack(env, n_stack=6)
agent = PPOAgent(env, model="models/ppo_car_racing_classic")
observation = env.reset()
# agent.train()

###################################

done = [False]
turn = 0.0
gas = 0.0
brake = 0.0
timestep = 0
tiles = 0
total_classic_reward = 0
total_energy_reward = 0

tiles_array = []
classic_reward_array = []
energy_reward_array = []
timesteps = []

while not False:
    action, _states = agent.model.predict(observation)
    observation, reward, done, info = env.step(action)
    total_classic_reward += reward

    ### GET ENERGY REWARD ###
    if action == 1:
        turn -= 0.2
    elif action == 2:
        turn += 0.2
    elif action == 3:
        gas += 0.2
    elif action == 4:
        gas -= 0.2
        brake += 0.2

    if turn > 0:
        turn -= 0.1
        turn = min(turn, 1.0)
    if turn < 0:
        turn += 0.1
        turn = max(turn, -1.0)
    gas -= 0.1
    brake -= 0.1
    gas = min(max(gas, 0.0), 1.0)
    brake = min(max(brake, 0.0), 1.0)

    def combine_rewards(reg_r, custom_r):
        return (custom_r < 0) * reg_r + (custom_r >= 0) * reg_r * custom_r

    total_energy_reward += combine_rewards(
        reward, 1 - computeLosses(action, observation, converted=(turn, gas, brake))
    )
    print(f"TOTAL ENERGY REWARD: {total_energy_reward}")
    print(f"TOTAL CLASSIC REWARD: {total_classic_reward}")
    classic_reward_array.append(total_classic_reward[0])
    energy_reward_array.append(total_energy_reward[0])
    timesteps.append(timestep)

    timestep += 1

    if done[0] == True:
        print("DONE")
        # tiles = 0
        # total_reward = 0

    if reward > 0:
        tiles += 1
        print(f"TILES: {tiles}")
    tiles_array.append(tiles)
    env.render()
    if done.any():
        observation = env.reset()

plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 2
plt.plot(tiles_array)
plt.xlabel("Timestep (t)")
plt.ylabel("Tiles Covered")
plt.title("Timestep vs Average Completion")

plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
plt.plot(timesteps, classic_reward_array, label="Classic", color="orange")
plt.plot(timesteps, energy_reward_array, label="Energy", color="green")
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward")
plt.title("Timestep vs Cumulative Reward")

plt.tight_layout()
plt.show()
input()
