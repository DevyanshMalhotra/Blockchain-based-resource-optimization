import simpy
import random
import numpy as np

class SimEnv:
    def __init__(self, config):
        self.env = simpy.Environment()
        self.config = config
        self.reset()

    def reset(self):
        self.env = simpy.Environment()
        n = self.config['num_peers']
        self.latencies = np.random.uniform(*self.config['latency_range'], (n, n))
        np.fill_diagonal(self.latencies, 0)
        self.mempool = [0 for _ in range(n)]
        self.time = 0.0
        return self._get_state()

    def _get_state(self):
        avg_lat = np.mean(self.latencies)
        avg_mem = np.mean(self.mempool)
        bw = self.config['bandwidth']
        return np.array([avg_lat, avg_mem, bw], dtype=np.float32)

    def step(self, action):
        n = self.config['num_peers']
        raw_mask = action[:n]
        if np.sum(raw_mask) < 1:
            mask = np.ones(n, dtype=bool)
        else:
            mask = raw_mask.astype(bool)
        block_size = action[-2]
        interval = action[-1] * self.config['max_interval']
        interval = max(interval, 1e-3)
        self.time += interval
        self.mempool = [max(0, m + random.randint(-5, 5)) for m in self.mempool]
        throughput = block_size / interval
        latencies = self.latencies[mask]
        avg_lat = float(np.mean(latencies)) if latencies.size else float(np.mean(self.latencies))
        orphan = random.random() * self.config['orphan_factor']
        reward = -self.config['alpha'] * avg_lat + self.config['beta'] * throughput - self.config['gamma'] * orphan
        next_state = self._get_state()
        done = self.time >= self.config['max_time']
        info = {'latency': avg_lat, 'throughput': throughput, 'orphan_rate': orphan}
        return next_state, reward, done, info
