import os
from abc import abstractmethod, ABC
import numpy as np
import math

from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy


def calculate_start_positions(board_width, board_height, angle):
    zombie_home_length = int(board_height - 2 * board_width * math.tan(math.pi * angle / 180))
    zombie_home_start_pos = int(
        board_height - zombie_home_length - board_width * math.tan(math.pi * angle / 180))  # m-n-b
    return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), board_width)


class Agent:

    def __init__(self, agent_type, config):
        main_info = config['MainInfo']
        self.interactive_mode = bool(main_info['interactive_mode'])
        self.display_width = int(main_info['display_width'])
        self.display_height = int(main_info['display_height'])
        self.num_train_episodes = int(main_info['num_train_episodes'])
        self.num_test_episodes = int(main_info['num_test_episodes'])
        self.zombies_per_episode = int(main_info['zombies_per_episode'])
        self.check_point = int(main_info['check_point'])
        self.light_size = int(main_info['light_size'])
        self.board_height = int(main_info['board_height'])
        self.board_width = int(main_info['board_width'])
        self.max_angle = float(main_info['max_angle'])
        self.max_velocity = int(main_info['max_velocity'])
        self.dt = float(main_info['dt'])
        self.max_hit_points = int(main_info['max_hit_points'])
        self.heal_points = float(main_info['heal_points'])
        self.end_learning_step = self.num_train_episodes * (self.zombies_per_episode + self.board_width)
        self.saved_model_path = os.path.join(main_info['checkpoint'], agent_type + "_player", self.__class__.__name__,
                                             "board_" + str(self.board_height) + "_" + str(self.board_width))
        self.load_model = eval(main_info['load_model'])

        self.agent_type = agent_type
        self.strategy = EpsilonGreedyStrategy(self.num_train_episodes, self.zombies_per_episode, self.board_width, config['StrategyInfo'])
        self.possible_actions = list(
            range(len(calculate_start_positions(self.board_width, self.board_height, self.max_angle)))) if agent_type == "zombie" else list(
            range(self.board_height * self.board_width))
        self.current_step = 0

    def get_neural_network(self):
        """
        :return: the NN used by the agent, for tensorboard purposes
        """
        return None

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self, state, action, next_state, reward):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def reset_start_pos(self):
        self.possible_actions = list(
            range(len(calculate_start_positions(self.board_width, self.board_height, self.max_angle)))) if self.agent_type == "zombie" else list(
            range(self.board_height * self.board_width))
