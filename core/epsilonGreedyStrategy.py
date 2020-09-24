import math


class EpsilonGreedyStrategy:
    def __init__(self, start, end, end_learning_step):
        self.start = start
        self.end = end
        self.end_learning_step = end_learning_step
        self.decay = math.log(end / start, math.exp(1)) / end_learning_step

    def get_exploration_rate(self, current_step):
        if current_step > self.end_learning_step:
            return self.end
        else:
            return self.start * math.exp(current_step * self.decay)
