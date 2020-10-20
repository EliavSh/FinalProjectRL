import random

from agents.agent import Agent
from runnable_scripts.Utils import get_config
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy


def update_variables():
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    CONST_ACTION = int(get_config("ConstAgentInfo")['const_action'])
    return BOARD_WIDTH, BOARD_HEIGHT, CONST_ACTION


class ConstantAgent(Agent):
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    CONST_ACTION = int(get_config("ConstAgentInfo")['const_action'])

    def __init__(self, device, agent_type):
        ConstantAgent.BOARD_WIDTH, ConstantAgent.BOARD_HEIGHT, ConstantAgent.CONST_ACTION = update_variables()

        super(ConstantAgent, self).__init__(EpsilonGreedyStrategy(), agent_type)
        self.current_step = 0
        self.possible_actions = list(range(ConstantAgent.BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(ConstantAgent.BOARD_HEIGHT * ConstantAgent.BOARD_WIDTH))
        self.constant_action = ConstantAgent.CONST_ACTION

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        return self.possible_actions[self.constant_action], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
