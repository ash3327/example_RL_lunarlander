# import pygame
import gymnasium as gym
import lib.lib_plot as plot

# env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
env = gym.make("LunarLander-v2", continuous=True)
observation, info = env.reset()

running = True
total_reward = 0

plot.init_plot('random')
REWARD_TABLE = 'final_reward'

epoch = 0
while running:
    # events = pygame.event.get()
    # for event in events:
    #     if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
    #         running = False

    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f'Session {epoch} ended {"successfully" if total_reward >= 200 else "in failure"} with reward: {total_reward}')
        plot.update_table(REWARD_TABLE, {'final_reward': total_reward}, epoch)
        observation, info = env.reset()
        total_reward = 0
        epoch += 1

env.close()