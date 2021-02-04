from agents.agent import Agent
from scipy.stats import truncnorm


def get_truncated_normal(mean, std, low, upp):
    return int(truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=std).rvs())


class GaussianAgent(Agent):
    def __init__(self, _, agent_type, config):
        super(GaussianAgent, self).__init__(agent_type, config)

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        if self.agent_type == 'zombie':
            mean = self.possible_actions[self.board_height // 2]
        else:
            mean = self.possible_actions[self.board_height * self.board_width // 2]

        if len(self.possible_actions) < 6:
            std = 1
        else:
            std = len(self.possible_actions) // 5

        return get_truncated_normal(mean, std, 0, len(self.possible_actions)), rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
