
"""
training mode:
    python lunarlander_qtrain.py
            -n --num_epochs : integer > 0

sample run:
    python lunarlander_qtrain.py -n 100000
"""

import gymnasium as gym
import lib.lib_plot as plot

# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_epochs', type=int, default=-1, help='Number of epochs of the random test.')
args = parser.parse_args()

num_epochs = args.num_epochs

# environment building
env = gym.make("LunarLander-v2", continuous=True)
observation, info = env.reset()

running = True
total_reward = 0

plot.init_plot('qtrain')
REWARD_TABLE = 'final_reward'

epoch = 0
while running and epoch != num_epochs:

    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f'Session {epoch} ended {"successfully" if total_reward>=200 else "in failure"} with reward: {total_reward}')
        plot.update_table(REWARD_TABLE, {'final_reward': total_reward}, epoch)
        observation, info = env.reset()
        total_reward = 0
        epoch += 1

env.close()