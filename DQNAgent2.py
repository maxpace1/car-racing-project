import gymnasium as gym
import time
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from rewardFunction import computeLosses


class DQNAgent2:
    def __init__(self, env, lr=0.00008, batch_size=64, train_freq=4, gamma=0.99, model=None):
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq

        if not model:
            self.model = DQN(
                "CnnPolicy",
                env,
                verbose=1,
                learning_rate=lr,
                batch_size=batch_size,
                gamma=gamma,
                buffer_size=100000, 
                train_freq=train_freq,
                target_update_interval=500
            )
        
        else:
            self.model = DQN.load(model)

    def train(self):
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path="./models/", name_prefix=f"dqn_car_racing_lr{self.lr}_tf{self.train_freq}_bs{self.batch_size}_g{self.gamma}_3"
        )
        self.model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        self.model.save(f"dqn_car_racing_{self.lr}_{self.batch_size}_{self.gamma}_2")

    # def predict(self, observation):
    #     action, _ = self.model.predict(observation)
    #     return action


def make_env(render_mode="rgb_array"):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=1.0,
        domain_randomize=True,
        continuous=False,
    )
    return env


def make_env_energy(render_mode="rgb_array", custom_reward=None):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=0.75,
        domain_randomize=True,
        continuous=False,
    )

    original_step = env.step

    def combine_rewards(reg_r, custom_r):
        return (custom_r < 0) * reg_r + (custom_r >= 0) * reg_r * custom_r

    env.turn = 0.0
    env.gas = 0.0
    env.brake = 0.0

    def new_step(action):
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

        out = original_step(action)
        if len(out) == 4:
            observation, reward, done, info = out
            if custom_reward is not None:
                reward = combine_rewards(reward, 1 - custom_reward(action, observation, converted=(env.turn, env.gas, env.brake)))
            return observation, reward, done, info
        elif len(out) == 5:
            observation, reward, done, info, _ = out
            if custom_reward is not None:
                reward = combine_rewards(reward, 1 - custom_reward(action, observation, converted=(env.turn, env.gas, env.brake)))
            return observation, reward, done, info, _
        else:
            return out

    env.step = new_step
    return env


def main():
    print("Initializing env...")

    mode = "test"

    if mode == "train":
        env = make_vec_env(env_id=make_env, n_envs=4)
        env = VecFrameStack(env, n_stack=4)

        lrs = [0.00008, 0.0001, 0.00015]
        batch_sizes = [64]
        tfs = [2, 4, 8]
        # gammas = [0.9, 0.99, 0.999]

        print("Starting tuning...")

        for a in lrs:
            for tf in tfs:
                # for c in gammas:
                print(f"\nTRAINING FOR LR {a} TRAIN FREQ {tf}\n")
                agent = DQNAgent2(env, a, 64, tf, model="None")
                observation = env.reset()
                agent.train()

    elif mode == "test":
        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        # env = DummyVecEnv([lambda: make_env_energy(render_mode="human", custom_reward=computeLosses)]) # Comment out for train
        env = VecFrameStack(env, n_stack=4)
    
        agent = DQNAgent2(env, model="models/dqn_car_racing_3618819_1")
        observation = env.reset()
        done = [False]

        tiles = 0
        tiles_array = []
        reward_array = []
        total_reward = 0
        while not done[0]:
            action, _states = agent.model.predict(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            print(f"TOTAL REWARD: {total_reward}")
            reward_array.append(total_reward[0])
            if done[0] == True:
                print("DONE")
                # tiles = 0
                # total_reward = 0

                time.sleep(10)
            if reward > 0:
                tiles += 1
                print(f"TILES: {tiles}")
            tiles_array.append(tiles)
            env.render()
            if done.any():
                observation = env.reset()
        
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 2
        plt.plot(tiles_array)
        plt.xlabel('Timestep (t)')
        plt.ylabel('Tiles Covered')
        plt.title("Timestep vs Average Completion")

        plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
        plt.plot(reward_array)
        plt.xlabel('Timestep (t)')
        plt.ylabel('Cumulative Reward')
        plt.title("Timestep vs Cumulative Reward")

        plt.tight_layout()
        plt.show()
        input()

if __name__ == "__main__":
    print("starting")
    main()

