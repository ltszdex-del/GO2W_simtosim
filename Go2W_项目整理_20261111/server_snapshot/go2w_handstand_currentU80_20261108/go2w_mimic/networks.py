"""教学版策略网络：教师用于 PPO，学生用于 DAgger 与 ONNX 部署。

固定的观测单位缩放写在 env.blocks 中，网络没有隐藏的运行均值归一化器。
所以导出时不会遗失归一化状态；部署必须严格遵守 interface.json。
"""
import math
import torch
from torch import nn
from torch.distributions import Normal
from .settings import TEACHER_DIM, STUDENT_DIM, PROPRIO_DIM, HISTORY, GOAL_DIM, ACTION_DIM


def mlp(sizes, output_gain=1.):
    modules=[]
    for i,(a,b) in enumerate(zip(sizes,sizes[1:])):
        layer=nn.Linear(a,b)
        nn.init.orthogonal_(layer.weight,gain=output_gain if i==len(sizes)-2 else math.sqrt(2))
        nn.init.zeros_(layer.bias)
        modules.append(layer)
        if i<len(sizes)-2: modules.append(nn.ELU())
    return nn.Sequential(*modules)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=mlp([TEACHER_DIM,256,128,64,ACTION_DIM],.01)
        self.critic=mlp([TEACHER_DIM,256,128,64,1])
        self.log_std=nn.Parameter(torch.full((ACTION_DIM,),math.log(.35)))

    def distribution(self, obs):
        return Normal(self.actor(obs),self.log_std.exp())

    def value(self,obs):
        return self.critic(obs).squeeze(-1)

    def forward(self,obs):
        return torch.tanh(self.actor(obs))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_encoder=mlp([HISTORY*PROPRIO_DIM,256,128,64])
        self.actor=mlp([64+GOAL_DIM,128,64,ACTION_DIM],.01)

    def forward(self,obs):
        history,goal=obs[:,:HISTORY*PROPRIO_DIM],obs[:,HISTORY*PROPRIO_DIM:]
        return torch.tanh(self.actor(torch.cat((self.history_encoder(history),goal),dim=-1)))
