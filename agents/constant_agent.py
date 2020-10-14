import random

from agents.agent import Agent
from runnable_scripts.Utils import get_config
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy

BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
CONST_ACTION = int(get_config("ConstAgentInfo")['const_action'])


class ConstantAgent(Agent):
    def __init__(self, device, agent_type):
        super(ConstantAgent, self).__init__(EpsilonGreedyStrategy(), agent_type)
        self.current_step = 0
        self.possible_actions = list(range(BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(BOARD_HEIGHT * BOARD_WIDTH))
        self.constant_action = CONST_ACTION

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        return self.possible_actions[self.constant_action], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
