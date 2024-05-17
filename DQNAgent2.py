import gymnasium as gym

from stable_baselines3 import DQN

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback


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
        lap_complete_percent=0.75,
        domain_randomize=True,
        continuous=False,
    )
    return env

def main():
    print("Initializing env...")
    # env = make_vec_env(env_id=make_env, n_envs=4)
    env = DummyVecEnv([lambda: make_env(render_mode="human")]) # Comment out for train
    env = VecFrameStack(env, n_stack=4)

    # lrs = [0.00008, 0.0001, 0.00015]
    # batch_sizes = [64]
    # tfs = [2, 4, 8]
    # # gammas = [0.9, 0.99, 0.999]

    # print("Starting tuning...")

    # for a in lrs:
    #     for tf in tfs:
    #         # for c in gammas:
    #         print(f"\nTRAINING FOR LR {a} TRAIN FREQ {tf}\n")
    #         agent = DQNAgent2(env, a, 64, tf, model="None")
    #         observation = env.reset()
    #         agent.train()
    
    agent = DQNAgent2(env, model="dqn_car_racing_3618819_1")
    observation = env.reset()
    while True:
        action, _states = agent.model.predict(observation)
        observation, reward, done, info = env.step(action)
        env.render()
        if done.any():
            observation = env.reset()

if __name__ == "__main__":
    print("starting")
    main()

