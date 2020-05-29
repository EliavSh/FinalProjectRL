import random


def init_q():
    pass


class ZombieMaster:
    def __init__(self, env):
        self.q = init_q
        self.alpha = 0.1
        # self.epsilon = 0.1 in the meantime the agent takes random action
        self.env = env
        self.current_action = 0

    def step(self):
        """
        for now the agent will take a random action (creates a zombie somewhere in the zombie-house)

        :return: action(position in the zombie house)
        """
        self.current_action = random.choice(self.env.start_positions)
        self.env.add_zombie(self.current_action)
        print("ZombieMaster action:", self.current_action)

    def learn(self, reward):
        """

        :return:
        """
        pass
