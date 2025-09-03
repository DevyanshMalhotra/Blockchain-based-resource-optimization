import yaml
from sim_env import SimEnv
from agent import DDPGAgent
from replay_buffer import ReplayBuffer
from metrics import Metrics
from tqdm import trange

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

env = SimEnv(cfg)
agent = DDPGAgent(3, cfg['num_peers'] + 2, cfg)
buffer = ReplayBuffer(cfg['buffer_size'])
metrics = Metrics()

for _ in trange(cfg['episodes']):
    s = env.reset()
    done = False
    while not done:
        action = agent.select_action(s)
        s2, r, done, info = env.step(action)
        buffer.add(s, action, r, s2, float(done))
        if len(buffer.mem) >= cfg['batch_size']:
            batch = buffer.sample(cfg['batch_size'])
            agent.learn(batch)
        metrics.record(info)
        s = s2

metrics.save('metrics.csv')
metrics.plot()
