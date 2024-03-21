import gymnasium as gym
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up display
screen = pygame.display.set_mode((640, 480))

is_pressed_left = False
is_pressed_right = False
is_pressed_space = False
is_pressed_shift = False
is_pressed_esc = False
steering_wheel = 0
gas = 0
break_system = 0

def check_events():
    global is_pressed_left, is_pressed_right, is_pressed_space, is_pressed_shift, is_pressed_esc
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(event.key)
            if event.key == pygame.K_LEFT:
                is_pressed_left = True
            elif event.key == pygame.K_RIGHT:
                is_pressed_right = True
            elif event.key == pygame.K_SPACE:
                is_pressed_space = True
            elif event.key == pygame.K_LSHIFT:
                is_pressed_shift = True
            elif event.key == pygame.K_ESCAPE:
                is_pressed_esc = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                is_pressed_left = False
            elif event.key == pygame.K_RIGHT:
                is_pressed_right = False
            elif event.key == pygame.K_SPACE:
                is_pressed_space = False
            elif event.key == pygame.K_LSHIFT:
                is_pressed_shift = False

def update_action():
    global steering_wheel
    global gas
    global break_system

    if is_pressed_left ^ is_pressed_right:
        if is_pressed_left:
            if steering_wheel > -1:
                steering_wheel -= 0.1
            else:
                steering_wheel = -1
        if is_pressed_right:
            if steering_wheel < 1:
                steering_wheel += 0.1
            else:
                steering_wheel = 1
    else:
        if abs(steering_wheel - 0) < 0.1:
            steering_wheel = 0
        elif steering_wheel > 0:
            steering_wheel -= 0.1
        elif steering_wheel < 0:
            steering_wheel += 0.1
    if is_pressed_space:
        if gas < 1:
            gas += 0.1
        else:
            gas = 1
    else:
        if gas > 0:
            gas -= 0.1
        else:
            gas = 0
    if is_pressed_shift:
        if break_system < 1:
            break_system += 0.1
        else:
            break_system = 1
    else:
        if break_system > 0:
            break_system -= 0.1
        else:
            break_system = 0

if __name__ == '__main__':
    env = gym.make("CarRacing-v2", render_mode="human")
    state = env.reset()

    counter = 0
    total_reward = 0
    while not is_pressed_esc:
        check_events()  # Check for key events
        update_action()
        action = [steering_wheel, gas, break_system]
        state, reward, terminated, truncated, info = env.step(action)
        counter += 1
        total_reward += reward
        print('State: {} Action:[{:+.1f}, {:+.1f}, {:+.1f}] Reward: {:.3f} Terminated: {} Truncated: {} Info: {}'.format(state, action[0], action[1], action[2], reward, terminated, truncated, info))
        if terminated:
            print("Restart game after {} timesteps. Total Reward: {}".format(counter, total_reward))
            counter = 0
            total_reward = 0
            state = env.reset()
            continue

        # Update Pygame display
        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()
