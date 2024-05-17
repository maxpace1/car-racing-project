import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import math


class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # Convolutional layers for image processing. Observations are 3 channel 96x86 pixel images
        # Each conv layer followed by activation function (ReLU)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Size of flattened convolutional layer output
        conv_out_size = int(torch.prod(torch.tensor(self.conv_layers(torch.zeros(1, *input_shape)).size())))
        
        # Fully connected layers to flatten conv output
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Send image through conv layers then fully connected layers
        x = x.permute(0, 3, 1, 2)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        return self.fc_layers(conv_out)
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQNAgent:
    def __init__(self, env, train):
        self.env = env

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        # Get the shape of state observations
        self.input_shape = (3, 96, 96)

        # Track the number of steps done for epsilon greedy decay
        self.steps_done = 0

        # MODEL HYPERPARAMETERS
        self.batch_size = 64            # number of transitions sampled from the replay buffer
        self.gamma = 0.99               # discount factor
        self.epsilon_start = 0.9        # epsilon-greedy parameter
        self.epsilon_end = 0.05         # starting value of epsilon
        self.epsilon_decay = 1000       # controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = 0.05                  # update rate of the target network
        self.learning_rate = 0.0001     # learning rate of the ``AdamW`` optimizer

        # use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trained_model_path = "dqn_model"

        # NNs
        if train:
            # Initialize new nns for training
            self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
            self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # Training optimizer
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        else:
            # Load trained model into the policy nn
            self.policy_net = DQN(self.input_shape, self.n_actions).load_state_dict(torch.load(self.trained_model_path))


    def predict(self, state):
        # Use random policy with epsilon probability, greedy o/w
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
    

    def optimize_model(self, memory):
        transitions = random.sample(memory, self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
       
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train(self):
        if torch.cuda.is_available():
            print("Using gpu")
            num_episodes = 50
        else:
            num_episodes = 50

        memory = deque([], maxlen=10000)
        episode_lengths = []
        episode_total_rewards = []

        for i in range(num_episodes):
            epidode_reward = 0
            negative_rewards = 0

            # Initialize the environment and get its state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in range(10000):
                action = self.predict(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                epidode_reward += reward
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Count steps with negative reward after first 100
                negative_rewards = negative_rewards + 1 if t > 100 and reward < 0 else 0

                # Store the transition in memory
                memory.append(Transition(state, action, next_state, reward))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(memory) >= self.batch_size:
                    self.optimize_model(memory)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                # If continually getting negative reward 25 times after the first 100 steps, terminate this episode
                if done or negative_rewards >= 25:
                    print(f"Episode {i} length: {t + 1}, reward: {epidode_reward}")
                    episode_lengths.append(t + 1)
                    episode_total_rewards.append(epidode_reward)
                    break
        
        plt.plot(episode_total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards During Training')
        plt.grid(True)
        plt.show()
        torch.save(self.policy_net.state_dict(self.trained_model_path))


def make_env(render_mode="rgb_array"):
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=True,
        continuous=False,
    )
    return env


if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=0.75, domain_randomize=True, continuous=False)
    # env = make_env()
    agent = DQNAgent(env, True)
    agent.train()
