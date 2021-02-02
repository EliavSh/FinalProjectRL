class Node:
    def __init__(self, state, possible_actions, board_width, zombies_per_episode):
        self.board_width = board_width
        self.zombies_per_episode = zombies_per_episode
        self.state = state
        self.wins = 0.0
        self.visits = 0.0
        self.parent = None
        self.children = {}.fromkeys(possible_actions)
        self.num_children = 0
        self.uct = 0.0
        self.is_terminal = False
        self.level = 0

    def add_child(self, state, action):
        new_child = Node(state, list(self.children.keys()), self.board_width, self.zombies_per_episode)
        self.children[action] = new_child
        self.num_children += 1
        assert self.num_children <= len(
            list(self.children.keys()))  # asserting that the number of children is never greater from the maximum possible children number
        new_child.parent = self
        new_child.level = self.level + 1
        assert new_child.level <= self.board_width + self.zombies_per_episode + 1
        if new_child.level >= self.board_width + self.zombies_per_episode:
            new_child.is_terminal = True
        return self.children[action]
