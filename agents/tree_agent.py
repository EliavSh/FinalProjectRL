from agents.agent import Agent


class TreeAgent(Agent):

    def __init__(self, strategy, agent_type):
        super().__init__(strategy, agent_type)

    def select_action(self, state):
        pass

    def learn(self, state, action, next_state, reward):
        pass
