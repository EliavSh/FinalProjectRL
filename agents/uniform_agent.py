import random
from agents.agent import Agent
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy


class UniformAgent(Agent):
    def __init__(self, device, agent_type, config):
        super(UniformAgent, self).__init__(agent_type, config)

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        return random.sample(self.possible_actions, 1)[0], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
