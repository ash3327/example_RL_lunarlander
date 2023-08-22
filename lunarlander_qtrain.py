
"""
training mode:
    python lunarlander_qtrain.py
            -n --num_epochs : integer > 0

sample run:
    python lunarlander_qtrain.py -n 100000
"""

# gym imports
import gymnasium as gym
import lib.lib_plot as plot

# DQN imports
from lib.lib_RL import Agent
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import adam_v2

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

"""
DQN / Deep Q-Network
"""
agent = Agent(
    env.action_space,
    model=Sequential(
        [
            tf.keras.Input(shape=env.observation_space.shape),
            Dense(1024, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(4)
        ]
    ),
    epsilon=.5,
    epsilon_min=0,
    epsilon_decay=1-1E-2,
    opt=adam_v2.Adam(learning_rate=1E-4)
)


"""
Training loop
"""
epoch = 0
step = 0
while running and epoch != num_epochs:

    action = agent.choose_action(observation)  # agent policy that uses the observation and info
    observation_, reward, terminated, truncated, info = env.step(action)

    agent.store_trainsition(observation, action, reward, observation_)

    if (step > 200) and(step % 5 == 0):
        agent.learn()

    observation = observation_
    total_reward += reward
    step += 1

    if terminated or truncated:
        # plotting
        print(f'Session {epoch} ended {"successfully" if total_reward>=200 else "in failure"} with reward: {total_reward}')
        plot.update_table(REWARD_TABLE, {'final_reward': total_reward}, epoch)
        observation, info = env.reset()
        total_reward = 0
        epoch += 1

env.close()