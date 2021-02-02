import random


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.memory_size] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
