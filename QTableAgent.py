import gymnasium as gym

import numpy as np

class QTableAgent:
    def __init__(self, env):
        self.env = env

        # Define Q-table dimensions
        print(f"Action space: {env.action_space}")
        self.n_actions = env.action_space.n
        self.state_shape = (16,)

        # Define Q-table
        self.Q_table = np.zeros((self.state_shape[0], self.n_actions))

        # Define hyperparameters
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # discount factor
        self.epsilon = 0.1  # exploration rate

    # Convert state to discretized representation
    def discretize_state(self, state):
        return tuple(int(s * 16) for s in state)
    
    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset()
            state = self.discretize_state(state)
            done = False

            while not done:
                # Choose action
                if np.random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.Q_table[state])

                # Perform action
                next_state, reward, done, _ = env.step(action)
                next_state = self.discretize_state(next_state)

                # Update Q-table
                best_next_action = np.argmax(self.Q_table[next_state])
                self.Q_table[state][action] += self.alpha * (reward + self.gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action])

                # Update state
                state = next_state

            return self.Q_table


if __name__ == "__main__":
    env = gym.make('CarRacing-v2')
    QTableAgent = QTableAgent(env)

    # Train the Q-table
    trained_Q_table = QTableAgent.train(env, episodes=1000)

    # Evaluate the trained Q-table
    total_reward = 0
    state = env.reset()
    state = QTableAgent.discretize_state(state)
    done = False

    while not done:
        action = np.argmax(trained_Q_table[state])
        state, reward, done, _ = env.step(action)
        state = QTableAgent.discretize_state(state)
        total_reward += reward

    print("Total reward:", total_reward)