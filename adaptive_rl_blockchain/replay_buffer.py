import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = []
        self.pos = 0

    def add(self, s, a, r, s2, done):
        data = (s, a, r, s2, done)
        if len(self.mem) < self.capacity:
            self.mem.append(data)
        else:
            self.mem[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done
