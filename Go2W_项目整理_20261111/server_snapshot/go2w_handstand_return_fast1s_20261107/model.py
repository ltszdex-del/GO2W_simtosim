import math
import torch
from torch import nn
from torch.distributions import Normal
from settings import ACTOR_DIM,ACTION_DIM,PPO

def mlp(sizes,gain=1.):
    layers=[]
    for i,(a,b) in enumerate(zip(sizes,sizes[1:])):
        linear=nn.Linear(a,b)
        nn.init.orthogonal_(linear.weight,gain if i==len(sizes)-2 else math.sqrt(2))
        nn.init.zeros_(linear.bias);layers.append(linear)
        if i<len(sizes)-2:layers.append(nn.ELU())
    return nn.Sequential(*layers)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=mlp([ACTOR_DIM,256,128,128,ACTION_DIM],.01)
        self.critic=mlp([ACTOR_DIM,256,128,128,1])
        self.log_std=nn.Parameter(torch.full((ACTION_DIM,),math.log(PPO['initial_std'])))
    def distribution(self,x):return Normal(self.actor(x),self.log_std.exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    def forward(self,x):return self.actor(x).tanh()
