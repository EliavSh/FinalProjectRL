from abc import abstractmethod

import numpy as np
import math

from runnable_scripts.Utils import get_config


def calculate_start_positions(BOARD_WIDTH, BOARD_HEIGHT, ANGLE):
    zombie_home_length = int(BOARD_HEIGHT - 2 * BOARD_WIDTH * math.tan(math.pi * ANGLE / 180))
    zombie_home_start_pos = int(
        BOARD_HEIGHT - zombie_home_length - BOARD_WIDTH * math.tan(math.pi * ANGLE / 180))  # m-n-b
    return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), BOARD_WIDTH)


class Agent:
    @staticmethod
    def update_variables():
        Agent.BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
        Agent.BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
        Agent.ANGLE = float(get_config("MainInfo")['max_angle'])

    # static field
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    ANGLE = float(get_config("MainInfo")['max_angle'])

    def __init__(self, strategy, agent_type):
        self.agent_type = agent_type
        self.strategy = strategy
        self.possible_actions = list(range(len(calculate_start_positions(Agent.BOARD_WIDTH, Agent.BOARD_HEIGHT,
                                                          Agent.ANGLE)))) if agent_type == "zombie" else list(
            range(Agent.BOARD_HEIGHT * Agent.BOARD_WIDTH))

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self, state, action, next_state, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reset_start_pos(self):
        Agent.update_variables()
        self.possible_actions = calculate_start_positions(Agent.BOARD_WIDTH, Agent.BOARD_HEIGHT, Agent.ANGLE)
