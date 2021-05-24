from .agent import Agent
from .constant_agent import ConstantAgent
from .double_constant_agent import DoubleConstantAgent
from .gaussian_agent import GaussianAgent
from .uniform_agent import UniformAgent


class CompositeAgent(Agent):
    """
    Uses the agents: Constant Double Gaussian and Uniform
    """

    def __init__(self, device, agent_type, config):
        super().__init__(agent_type, config)

        self.const_agent = ConstantAgent(device, agent_type, config)
        self.double_agent = DoubleConstantAgent(device, agent_type, config)
        self.gaussian_agent = GaussianAgent(device, agent_type, config)
        self.uniform_agent = UniformAgent(device, agent_type, config)

    def select_action(self, state):
        # we should take actions with them all for incrementing their current_step
        const_action = self.const_agent.select_action(state)
        double_action = self.double_agent.select_action(state)
        gaussian_action = self.gaussian_agent.select_action(state)
        uniform_action = self.uniform_agent.select_action(state)

        actions = [const_action, double_action, gaussian_action, uniform_action]

        self.current_step += 1
        num_of_steps_in_episode = self.zombies_per_episode + self.board_width

        return actions[(self.current_step % (num_of_steps_in_episode * len(actions))) // num_of_steps_in_episode]

    def learn(self, state, action, next_state, reward):
        pass

    def reset(self):
        pass
