
# gym imports
import gymnasium as gym
import lib.lib_plot as plot


class Agent:
    def __init__(self, action_space: gym.Space, model, epsilon: float, epsilon_min: float, epsilon_decay: float, opt):
        """
        model: the Tensorflow model.
        opt: the tensorflow optimizer.
        """
        self.action_space = action_space
        self.model = model
        self.epsilon =  epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.opt = opt
        model.build()
        model.summary(print_fn=lambda contents: plot.write('model_structure.txt', contents))

    def choose_action(self, observation):
        return self.action_space.sample()  # random

    def store_trainsition(self, observation, action, reward, observation_):
        pass

    def learn(self):
        pass

