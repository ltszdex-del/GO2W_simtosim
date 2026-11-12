"""Only transitions actually controlled by the return actor contribute to PPO."""
import torch

def masked_gae(rewards,dones,values,last_value,controlled,gamma=.995,lam=.95):
    advantage=torch.zeros_like(rewards);carry=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        following=last_value if t==len(rewards)-1 else values[t+1]
        alive=(~dones[t]).float()
        delta=rewards[t]+gamma*following*alive-values[t]
        carry=(delta+gamma*lam*alive*carry)*controlled[t]
        advantage[t]=carry
    return advantage,advantage+values
