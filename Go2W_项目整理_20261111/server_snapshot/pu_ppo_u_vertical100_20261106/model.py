import math
import torch
from torch import nn
from torch.distributions import Normal

def mlp(sizes,gain=1):
    layers=[]
    for i,(a,b) in enumerate(zip(sizes,sizes[1:])):
        layer=nn.Linear(a,b)
        nn.init.orthogonal_(layer.weight,gain if i==len(sizes)-2 else math.sqrt(2))
        nn.init.zeros_(layer.bias); layers.append(layer)
        if i<len(sizes)-2: layers.append(nn.ELU())
    return nn.Sequential(*layers)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=mlp([216,256,128,64,12],.01)
        self.critic=mlp([216,256,128,64,1])
        self.log_std=nn.Parameter(torch.full((12,),math.log(.35)))
    def distribution(self,x): return Normal(self.actor(x),self.log_std.exp())
    def value(self,x): return self.critic(x).squeeze(-1)
    def forward(self,x): return self.actor(x).tanh()
