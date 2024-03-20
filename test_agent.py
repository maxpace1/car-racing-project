import gymnasium as gym
import sys
from DQNAgent import DQNAgent
from PPOAgent import PPOAgent
from QTableAgent import QTableAgent

class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def predict(self, observation):
        # Return action sampled randomly from action space
        return env.action_space.sample()


if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    # normal reset
    observation = env.reset()

    # Determine the agent from user argument
    agent_type = sys.argv[1]
    if agent_type == "dqn":
        agent = DQNAgent(env)
    elif agent_type == "ppo":
        agent = PPOAgent(env)
    elif agent_type == "qtable":
        agent = QTableAgent(env)
    else:
        agent = RandomAgent(env)

    total_reward = 0

    # Evaluate agent over 1000 interactions with the environment
    for i in range(1000):
        env.render()
        # Get action from agent
        action = agent.predict(observation)
        # Perform action on environment, observe state + reward
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print("terminating")
            observation = env.reset()

    print(f"Total Reward: {total_reward}")
    env.close()