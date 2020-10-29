from runnable_scripts.Utils import get_config


class Node:
    @staticmethod
    def update_variables():
        Node.BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
        Node.ZOMBIES_PER_EPISODE = int(get_config("MainInfo")['zombies_per_episode'])

    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    ZOMBIES_PER_EPISODE = int(get_config("MainInfo")['zombies_per_episode'])

    def __init__(self, state, possible_actions):
        self.state = state
        self.wins = 0.0
        self.visits = 0.0
        self.parent = None
        self.children = {}.fromkeys(possible_actions)
        self.num_children = 0
        self.uct = 0.0
        self.is_terminal = False
        self.level = 0

    def set_weight(self, weight):
        self.weight = weight

    def add_child(self, state, action):
        # add a child if we never did it before at all
        if self.children[action] is None:
            new_child = Node(state, list(self.children.keys()))
            self.children[action] = new_child
            self.num_children += 1
            new_child.parent = self
            new_child.level = self.level + 1
            if new_child.level >= Node.BOARD_WIDTH + Node.ZOMBIES_PER_EPISODE:
                new_child.is_terminal = True
        return self.children[action]

    def is_equal(self, node):
        if self.state == node.state:
            return True
        else:
            return False
