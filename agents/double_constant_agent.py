from agents.agent import Agent
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy


class DoubleConstantAgent(Agent):
    def __init__(self, device, agent_type, config):
        super(DoubleConstantAgent, self).__init__(agent_type, config)

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        if self.current_step % 2 == 0:
            return self.possible_actions[0], rate, self.current_step
        else:
            if self.agent_type == 'zombie':
                return self.possible_actions[self.board_height // 2], rate, self.current_step
            else:
                return self.possible_actions[self.board_height * self.board_width // 2], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
