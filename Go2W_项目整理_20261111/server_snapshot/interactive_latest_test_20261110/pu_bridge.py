"""Engine constructor used by private, import-rewritten PU policy snapshots.

Only scene construction is replaced. Policy frame/control/success methods stay original.
No reset or in-episode state writes are exposed here.
"""
import torch

class EmbeddedEnvBase:
    def __init__(self,cfg):
        self.cfg=cfg
        env=cfg.embedded_env
        self.device=env.device;self.num_envs=env.num_envs;self.step_dt=env.step_dt
        self.robot=env.robot;self.contacts=env.contacts;self.scene=env.scene
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.max_episode_length=10**9;self.extras={}

    def _reset_idx(self,*_):raise RuntimeError('Embedded PU policies must never reset physical state')

