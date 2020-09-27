import random
from runnable_scripts.Utils import get_config


class ReplayMemory:
    def __init__(self):
        self.memory_info = get_config('DdqnAgentInfo')
        self.capacity = int(self.memory_info['memory_size'])
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
