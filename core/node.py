from runnable_scripts.Utils import get_config

BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
ZOMBIES_PER_EPISODE = int(get_config("MainInfo")['zombies_per_episode'])


class Node:
    def __init__(self, state, possible_actions, simulated_node=False):
        self.state = state
        self.simulated_node = simulated_node
        self.wins = 0.0
        self.visits = 0.0
        self.ressq = 0.0
        self.parent = None
        self.children = {}.fromkeys(possible_actions)
        self.num_children = 0
        self.sputc = 0.0
        self.is_terminal = False
        self.level = 0

    def set_weight(self, weight):
        self.weight = weight

    def add_child(self, state, action, simulated_child=False):
        # add a child if we never did it before at all
        if self.children[action] is None:
            new_child = Node(state, list(self.children.keys()))
            self.children[action] = new_child
            self.num_children += 1
            new_child.parent = self
            new_child.level = self.level + 1
            if new_child.level >= BOARD_WIDTH:
                new_child.is_terminal = True
            new_child.simulated_node = simulated_child
        # and if there was a simulated child there, update it with the current state
        # moving from simulated to real node
        elif len(self.children[action].state) == 0 and not simulated_child:
            self.children[action].state = state
            self.children[action].simulated_node = False
            if self.children[action].level >= BOARD_WIDTH:
                self.children[action].is_terminal = True
        return self.children[action]

    def is_equal(self, node):
        if self.state == node.state:
            return True
        else:
            return False
