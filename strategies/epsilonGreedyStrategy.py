import math
from strategies.strategy import Strategy


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, num_train_episodes, zombies_per_episode, board_width, strategy_info):
        # load main info
        self.end_learning_step = num_train_episodes * (zombies_per_episode + board_width + 2) or 1
        # load strategy info
        self.start = float(strategy_info['eps_start'])
        self.end = math.exp(int(strategy_info['eps_end']))
        self.decay = math.log(self.end / self.start, math.exp(1)) / self.end_learning_step

    def get_exploration_rate(self, current_step):
        if current_step >= self.end_learning_step:
            return 0
        else:
            return self.start * math.exp(current_step * self.decay)
