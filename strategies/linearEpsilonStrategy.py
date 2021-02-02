import math

from strategies.strategy import Strategy


class LinearEpsilonStrategy(Strategy):
    def __init__(self, num_train_episodes, zombies_per_episode, board_width, _):
        self.end_learning_step = num_train_episodes * (zombies_per_episode + board_width + 2)

    def get_exploration_rate(self, current_step):
        return (self.end_learning_step - current_step) / self.end_learning_step if current_step < self.end_learning_step else 0
