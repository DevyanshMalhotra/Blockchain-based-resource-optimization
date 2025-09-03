import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(s_dim, 64), nn.ReLU(), nn.Linear(64, a_dim), nn.Sigmoid())

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(s_dim + a_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, a):
        return self.fc(torch.cat([s, a], dim=1))

class DDPGAgent:
    def __init__(self, s_dim, a_dim, cfg):
        self.actor = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.target_actor = Actor(s_dim, a_dim)
        self.target_critic = Critic(s_dim, a_dim)
        self._sync_targets()
        self.a_opt = optim.Adam(self.actor.parameters(), lr=cfg['lr_actor'])
        self.c_opt = optim.Adam(self.critic.parameters(), lr=cfg['lr_critic'])
        self.noise_scale = cfg['noise_scale']
        self.cfg = cfg

    def _sync_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        s = torch.from_numpy(state).unsqueeze(0)
        a = self.actor(s).detach().numpy().flatten()
        a = np.clip(a + np.random.normal(scale=self.noise_scale, size=a.shape), 0, 1)
        mask = (a[:self.cfg['num_peers']] > 1.0 / self.cfg['num_peers']).astype(float)
        block_size = a[-2] * self.cfg['max_block_size']
        interval = a[-1] * self.cfg['max_interval']
        return np.concatenate([mask, [block_size, interval]])

    def learn(self, batch):
        s, a, r, s2, d = batch
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        r = torch.from_numpy(r).unsqueeze(1).float()
        s2 = torch.from_numpy(s2).float()
        d = torch.from_numpy(d).unsqueeze(1).float()
        q1 = self.critic(s, a)
        with torch.no_grad():
            a2 = self.target_actor(s2)
            q2 = self.target_critic(s2, a2)
            target = r + self.cfg['gamma'] * (1 - d) * q2
        loss_c = nn.MSELoss()(q1, target)
        self.c_opt.zero_grad()
        loss_c.backward()
        self.c_opt.step()
        a_pred = self.actor(s)
        loss_a = -self.critic(s, a_pred).mean()
        self.a_opt.zero_grad()
        loss_a.backward()
        self.a_opt.step()
        self._update_targets()

    def _update_targets(self):
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(tp.data * (1 - self.cfg['tau']) + p.data * self.cfg['tau'])
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.cfg['tau']) + p.data * self.cfg['tau'])
