import yaml
import numpy as np
from sim_env import SimEnv
from agent import DDPGAgent
from metrics import ComparativeMetrics

def static_policy(state, cfg):
    mask = np.ones(cfg['num_peers'])
    block_size = cfg['max_block_size']
    interval = cfg['max_interval']
    return np.concatenate([mask, [block_size, interval]])

def rule_based_policy(state, cfg):
    num_peers = cfg['num_peers']
    avg_lat, avg_mem, _ = state
    lat = np.random.uniform(*cfg['latency_range'], num_peers)
    idx = np.argsort(lat)[: num_peers // 2]
    mask = np.zeros(num_peers)
    mask[idx] = 1
    block_size = min(cfg['max_block_size'], avg_mem)
    interval = cfg['max_interval'] * (1 if avg_mem > cfg['mempool_threshold'] else 0.5)
    return np.concatenate([mask, [block_size, interval]])

def evaluate(policy_fn, agent, env, cfg, episodes=5):
    m = {'latency': [], 'throughput': [], 'orphan_rate': []}
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if policy_fn == 'rl':
                a = agent.select_action(s)
            else:
                a = policy_fn(s, cfg)
            s2, r, done, info = env.step(a)
            m['latency'].append(info['latency'])
            m['throughput'].append(info['throughput'])
            m['orphan_rate'].append(info['orphan_rate'])
            s = s2
    return m

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

env = SimEnv(cfg)
agent = DDPGAgent(3, cfg['num_peers'] + 2, cfg)
comp = ComparativeMetrics()
comp.add('Static', evaluate(static_policy, None, env, cfg))
comp.add('RuleBased', evaluate(rule_based_policy, None, env, cfg))
comp.add('RL-Orch', evaluate('rl', agent, env, cfg))
comp.plot_comparison()
