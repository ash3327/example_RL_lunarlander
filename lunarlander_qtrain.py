
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

# plotting
plot.init_plot('qtrain')
REWARD_TABLE = 'final_reward'

# DQN / Deep Q-Network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import adam_v2

input_shape = env.observation_space.shape
model = Sequential(
    [
        tf.keras.Input(shape=input_shape),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(4)
    ]
)

epoch = 0
while running and epoch != num_epochs:

    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        # plotting
        print(f'Session {epoch} ended {"successfully" if total_reward>=200 else "in failure"} with reward: {total_reward}')
        plot.update_table(REWARD_TABLE, {'final_reward': total_reward}, epoch)
        observation, info = env.reset()
        total_reward = 0
        epoch += 1

env.close()