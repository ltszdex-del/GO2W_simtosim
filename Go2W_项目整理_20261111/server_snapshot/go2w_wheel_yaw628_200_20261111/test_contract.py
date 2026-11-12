import torch
from schedule import Commands,Schedule
from learning import Policy,advantages
from config import ACTOR_DIM,CRITIC_DIM

c=Commands(4,'cpu');ids=torch.arange(4);c.reset(ids,.7)
axes=torch.tensor([[1.,0,0],[-1.,0,0],[0,1.,0],[1.,1.,0]])
for _ in range(200):c.tick(axes,.02)
assert (c.velocity[:,:2].norm(dim=-1)<=1.500001).all()
assert c.velocity[0,0]>1.49 and c.velocity[1,0]<-1.49 and c.velocity[2,1]>1.49
assert c.heading.eq(.7).all() and c.velocity[:,2].eq(0).all()
for _ in range(50):c.tick(torch.zeros_like(axes),.02)
assert c.velocity.norm(dim=-1).max()<.011
c.reset(ids)
for _ in range(30):c.tick(axes,.02)
assert c.velocity[0,0]>.5
for _ in range(60):c.tick(-axes,.02)
assert c.velocity[0,0]<-.5
s=Schedule(1024,'cpu',1);s.reset(torch.arange(1024))
s.reset(torch.arange(1024,dtype=torch.int32))
v=s.tick(1,torch.ones(1024),.02)
assert v[:,:2].abs().sum(-1).eq(1).all()
assert all((v[:,j]==sign).sum()==256 for j,sign in [(0,1),(0,-1),(1,1),(1,-1)])
assert s.tick(5,torch.zeros(1024),.02).eq(0).all()
assert s.tick(5,torch.full((1024,),10.),.02).eq(0).all()
m=Policy();out=m(torch.zeros(3,ACTOR_DIM))
assert out.shape==(3,12) and torch.isfinite(out).all()
# Failure blocks GAE across resets, normal timeout is compensated by caller.
adv,_=advantages(torch.tensor([[1.],[2.]]),torch.tensor([[True],[False]]),
                 torch.zeros(2,1),torch.zeros(1),.99,.95)
assert float(adv[0])==1.
print('PASS: speed norm, per-axis damping, reversal, fixed yaw, four-direction balance, release, actor, reset GAE')
