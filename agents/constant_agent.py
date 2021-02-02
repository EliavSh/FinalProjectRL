from agents.agent import Agent


class ConstantAgent(Agent):
    def __init__(self, device, agent_type, config):
        super(ConstantAgent, self).__init__(agent_type, config)

        # load values from config
        constant_agent_config = config['ConstAgentInfo']
        self.constant_action = int(constant_agent_config['const_action'])

    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1
        return self.possible_actions[self.constant_action], rate, self.current_step

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
