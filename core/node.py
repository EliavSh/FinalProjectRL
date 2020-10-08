class Node:
    def __init__(self, state, parent_node=None):
        self.state = state
        self.wins = 0.0
        self.visits = 0.0
        self.ressq = 0.0
        self.parent = None
        self.children = []
        self.sputc = 0.0
        self.level = 0 if not parent_node else parent_node.level + 1

    def set_weight(self, weight):
        self.weight = weight

    def append_child(self, child):
        self.children.append(child)
        child.parent = self
        child.level = self.level + 1

    def is_equal(self, node):
        if self.state == node.state:
            return True
        else:
            return False
