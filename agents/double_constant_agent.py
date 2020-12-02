import random

from agents.agent import Agent
from runnable_scripts.Utils import get_config
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy


def update_variables():
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    return BOARD_WIDTH, BOARD_HEIGHT


class DoubleConstantAgent(Agent):
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])

    def __init__(self, device, agent_type):
        DoubleConstantAgent.BOARD_WIDTH, DoubleConstantAgent.BOARD_HEIGHT = update_variables()

        super(DoubleConstantAgent, self).__init__(EpsilonGreedyStrategy(), agent_type)
        self.current_step = 0

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        if self.current_step % 2 == 0:
            return self.possible_actions[0], rate, self.current_step
        else:
            if self.agent_type == 'zombie':
                return self.possible_actions[DoubleConstantAgent.BOARD_HEIGHT // 2], rate, self.current_step
            else:
                return self.possible_actions[DoubleConstantAgent.BOARD_HEIGHT * DoubleConstantAgent.BOARD_WIDTH // 2], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
