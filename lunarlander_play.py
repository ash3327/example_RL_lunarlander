
"""
human mode:
    python lunarlander_play.py
"""

import pygame
import numpy as np
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
observation, info = env.reset()

running = True
main = 0
lateral = 0

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # agent policy that uses the observation and info
            #lateral *= .9
            main *= .9
            if event.key == pygame.K_LEFT:
                if -.5 < lateral <= 0:
                    lateral = -.5
                elif 0 < lateral < .5:
                    lateral = 0
                else:
                    lateral -= .1
            if event.key == pygame.K_RIGHT:
                if 0 <= lateral < .5:
                    lateral = .5
                elif -.5 < lateral < 0:
                    lateral = 0
                else:
                    lateral += .1
            if event.key == pygame.K_UP:
                main += .3
            if event.key == pygame.K_DOWN:
                main -= .3
            if event.key == pygame.K_ESCAPE:
                running = False
    observation, reward, terminated, truncated, info \
        = env.step(np.array([np.clip(main, -1, 1), np.clip(lateral, -1, 1)]))

    if terminated or truncated:
        main = 0
        lateral = 0
        observation, info = env.reset()

env.close()