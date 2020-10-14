from abc import abstractmethod


class Strategy:

    @abstractmethod
    def get_exploration_rate(self, current_step):
        pass
