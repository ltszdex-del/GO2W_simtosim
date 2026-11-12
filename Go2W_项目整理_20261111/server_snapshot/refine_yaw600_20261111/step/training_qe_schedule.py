import torch
from qe_schedule import MixedSchedule
class TrainingSchedule(MixedSchedule):
    def __init__(self,n,device,seed,qe_stage=1):
        super().__init__(n,device,seed,qe_stage)
        ids=torch.arange(n,device=device)
        self.combo_mask=self.qe_mask & (ids%4>=2)
        self.pure_mask=self.qe_mask & ~self.combo_mask
        self.turn_progress=0.
    def tick(self,stage,age,dt):
        original=self.qe_stage;self.qe_stage=2
        axes=super().tick(stage,age,dt);self.qe_stage=original
        axes[self.pure_mask,:2]=0.
        axes[self.combo_mask,:2]*=.35+.65*self.turn_progress
        axes[self.pure_mask,2]=self.yaw_sign[self.pure_mask]*(self.yaw_value[self.pure_mask]!=0)
        axes[self.combo_mask,2]*=.5+.5*self.turn_progress
        axes[(age<.5)|(age>=9.)]=0.
        return axes
