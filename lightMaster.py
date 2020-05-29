import random


def init_q():
    pass


class LightMaster:
    def __init__(self, env):
        self.q = init_q
        self.alpha = 0.1
        # self.epsilon = 0.1 in the meantime the agent takes random action
        self.env = env
        self.current_action = 0

    def step(self):
        """
        for now the agent will take a random action - points the light somewhere in the grid
        :return:
        """
        self.current_action = random.randint(0, self.env.grid.get_height() * self.env.grid.get_width())
        print("LightMaster action:", self.current_action)
        return self.current_action

    def learn(self, reward):
        """

        :return:
        """
        pass
