import math
from strategies.strategy import Strategy
from runnable_scripts.Utils import get_config


class EpsilonGreedyStrategy(Strategy):
    def __init__(self):
        # load main info
        main_info = get_config("MainInfo")
        self.end_learning_step = int(main_info['num_train_episodes']) * (int(main_info['zombies_per_episode']) + int(main_info['board_width']) + 2)
        # load strategy info
        strategy_info = get_config('StrategyInfo')
        self.start = float(strategy_info['eps_start'])
        self.end = math.exp(int(strategy_info['eps_end']))
        self.decay = math.log(self.end / self.start, math.exp(1)) / self.end_learning_step

    def get_exploration_rate(self, current_step):
        if current_step > self.end_learning_step:
            return self.end
        else:
            return self.start * math.exp(current_step * self.decay)
