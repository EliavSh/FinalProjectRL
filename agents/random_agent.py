import random

from agents.agent import Agent
from runnable_scripts.Utils import get_config
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy

BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])


class RandomAgent(Agent):
    def __init__(self, device, agent_type):
        super(RandomAgent, self).__init__(EpsilonGreedyStrategy(), agent_type)
        self.current_step = 0
        self.possible_actions = list(range(BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(BOARD_HEIGHT * BOARD_WIDTH))

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        return random.sample(self.possible_actions, 1)[0], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
