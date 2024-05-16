import gymnasium as gym
import numpy as np
import cv2
import time
import csv
import cur_Q_table
import rewardFunction

class QTableAgent:
    def __init__(self, env):
        self.env = env

        # Define state and action space dimensions
        self.state_shape = (5,6)
        self.n_actions = env.action_space.n

        # Define Q-table as dictionary per: 
        #    {< State>:[Action0, Action1, Action2, Action3, Action4]}
        self.Q_table = {}

        # Define parameters
        self.alpha = 0.9  # learning rate
        self.gamma = 0.99  # discount factor
        self.epsilon = 0.1  # exploration rate

        # For detecting stationary
        self.prevOriginal = None

    def convert_downsized_binary(self, original):
        """
        Convert original state representation to downsized binary representation.
            self: QTableAgent Object
            original: 96x96 color image state
        """

        # Busy wait for state rendering animation before capture
        while original is None:
                original, _, _, _, _ = env.step(0)
        
        # Load original state rep, slice and downsize to 5x6
        original = original[:80]
        downsized = cv2.resize(original, (6, 5), interpolation=cv2.INTER_AREA)
            
        # Threshold downsized to convert to binary rep
        greyscale = cv2.cvtColor(downsized, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(greyscale, 150, 255, cv2.THRESH_BINARY)
        
        # Store pre and post processing for debugging
        # cv2.imwrite("Original Image.jpg", original)
        # cv2.imwrite("Binary Image.jpg", binary)

        # Invert binary, making roads 1, force to float
        binary = np.int64(binary/255)

        # Encode state as a binary number, converted to decimal
        # Following 2^ mapping:
        # | 0  1  2  3  4  5  |
        # | 6  7  8  9  10 11 |
        # | 12 13 14 15 16 17 |
        # | 18 19 20 21 22 23 |
        # | 24 25 26 27 28 29 |

        # For testing map 
        # test = np.zeros(self.state_shape)

        encoding = 0
        for row in range(binary.shape[0]):
            for col in range(binary.shape[1]):

                encoding += int(not binary[row][col]) * 2**((row * (binary.shape[1])) + col)

                # For testing map
                # encoding = 2**((row * (binary.shape[1])) + col)
                # test[row][col] = encoding

        # print(f"{test} is the test matrix")
        
        # Set as state
        state = encoding
        #print(f"{state} is the state")

        # Experimental: Reward lower two blocks being road
        # bonus_reward = 0
        # if not binary[4][2] and not binary[4][3] and binary[4][0] and binary[4][5]: bonus_reward = 10

        # If similar (not necessarily identical due to stationary steering rotation) original state representation consecutively, means not moving. Mark as negative state.
        if self.prevOriginal is not None:
            if abs(np.sum(original) - np.sum(self.prevOriginal)) < 200: 
                state = -int(state)
                self.fixed += 1
                
                # Experimental: Penalize stationary
                # bonus_reward -= 20
            else:
                self.fixed = 0

        # Experimental: Bonus reward
        # print(f"{bonus_reward}  is the bonus reward")

        # Store original for use in stationary testing
        self.prevOriginal = original
        return (state)

    def train(self, env, episodes=10000, iterations=300000):
        for episode in range(1, episodes+1):
            env.reset()

            # Tracking stationary
            self.prevOriginal = None
            self.fixed = 0

            # Speed up training by terminating if off-track too long
            offTrackCycles = 0

            # Track rewards
            total_reward = 0
            actual_reward = 0

            # Track tiles touched
            tiles_crossed = 0

            # Retrieve starting downsized binary state representation
            state = self.convert_downsized_binary(env.render())

            # If at a state not visited before, create it
            if self.Q_table.get(state) is None:
                self.Q_table[state] = [0,0,0,0,0]

            # Decaying epsilon
            if (episode % 1000 == 0) and self.epsilon > 0.1: self.epsilon -= 0.1
        
            for t in range(iterations):

                # To check if any actions yield positive rewards
                # for possibleAction in range(5):
                #     _, reward, _, _, _ = env.step(possibleAction)
                #     print(f"{possibleAction} yields reward {reward}")

                # Sample randomly from actions if under epsilon
                if np.random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.Q_table[state])

                # Perform action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Update if tile crossed
                if reward > 0: tiles_crossed += 1

                # Retrieve current downsized binary state representation
                next_state  = self.convert_downsized_binary(next_state)

                # Punish having no track visibility severely
                if next_state == 0:
                    offTrackCycles += 1 
                    reward -= 100
                else: offTrackCycles = 0

                # If next state not seen before, create it
                if self.Q_table.get(next_state) is None:
                    self.Q_table[next_state] = [0,0,0,0,0]

                # Apply Q-function, update Q-table with next best action
                best_next_action = np.argmax(self.Q_table[next_state])
                self.Q_table[state][action] += self.alpha * ((reward) + self.gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action])

                actual_reward += reward

                # Experiment: Bonus reward
                #total_reward += reward + bonus_reward
                
                # Update state
                state = next_state

                # Terminate episode conditions
                if terminated or offTrackCycles > 5 or self.fixed > 25 or actual_reward < -300: break

            # Print results from this episode
            print(f"{episode} Reward: {total_reward} Tiles Crossed: {tiles_crossed} Q-table: {self.Q_table}")

            # Store latest QTable
            with open("q_table.txt", 'w') as file:
                # Write the q table to the file
                file.write("episode " + str(episode) + ": " + str(self.Q_table))

            # Store rewards and tile completion
            with open("rewardsOverTraining.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, tiles_crossed, actual_reward])

        return self.Q_table

if __name__ == "__main__":
    env = gym.make('CarRacing-v2', render_mode="human", lap_complete_percent=0.5, continuous=False)
    QTableAgent = QTableAgent(env)

    # Clear rewards over training CSV file
    with open("rewardsOverTraining.csv", 'w') as file:
        pass

    # Train the Q-table
    # trained_Q_table = QTableAgent.train(env, episodes=1000, iterations=100000)
    # print(f"{trained_Q_table} is the final Q table")

    trained_Q_table = cur_Q_table.Q_table

    # Clear time/completion mapping file
    with open("time_completion.csv", 'w') as file:
        pass

    # Evaluate the trained Q-table
    cumulative_reward = 0
    tiles_crossed = 0
    env.reset()
    state = QTableAgent.convert_downsized_binary(env.render())
    done = False

    # init timestep
    t = 1

    while not done and t < 300000:
        try:
            # Handle yet unseen states
            if trained_Q_table.get(state) is None: action = 3

            # Act deterministically otherwise
            else: action = np.argmax(trained_Q_table[state])
            state, reward, done, _, info = env.step(action)
            state = QTableAgent.convert_downsized_binary(state)

            # Update rewards and tile crossing
            if reward > 0: tiles_crossed += 1
            cumulative_reward += reward

            with open("time_completion.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, tiles_crossed, cumulative_reward])

            t += 1
        except KeyboardInterrupt:
            break

    print("Total reward:", cumulative_reward, "Tiles crossed:", tiles_crossed)