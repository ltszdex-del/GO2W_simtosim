import torch
def reference_phase(age,period):
    return age*(2*torch.pi/period)

def sample_period(n,device,training,evaluation_period=.28):
    return .23+.10*torch.rand(n,device=device) if training else torch.full((n,),evaluation_period,device=device)
