"""Distilled H actor with the verified teacher's motion, guards and local fixes."""
import hashlib
from pathlib import Path
import torch
from handstand_adapter import HandstandAdapter

class HandstandStudentAdapter(HandstandAdapter):
    def __init__(self,spec,device):
        teacher_spec={**spec,'checkpoint':spec['teacher_checkpoint']}
        super().__init__(teacher_spec,device)
        teacher_digest=hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        assert teacher_digest==spec['teacher_sha256'],'Student teacher mismatch'
        self.teacher_path=self.path
        self.spec=spec;self.path=spec['checkpoint']
        self.ck=torch.load(self.path,map_location='cpu',weights_only=False)
        assert self.ck['kind']=='student'
        assert self.ck['interface']==self.saved,'Student observation/controller contract mismatch'
        assert (self.saved['history'],self.saved['proprio_dim'],self.saved['goal_dim'],
                self.saved['student_observations'])==(50,52,57,2657)
        self.manifest.update(observation_order=['50 x proprio, oldest to newest']+self.saved['goal_order'],
            history=50,actor_input_dim=2657,actor_uses_privileged_state=False,
            policy_kind='student',teacher_sha256=teacher_digest)

    def load(self):
        self.model=self.networks.Student().to(self.device)
        self.model.load_state_dict(self.ck['model'],strict=True);self.model.eval()
        self.motion=self.module.Motion(self.spec['motion'],self.device)

    def observe(self,state):
        observation=self.ctx._get_observations()['student']
        assert observation.shape==(1,2657)
        return observation

def create(spec,device):return HandstandStudentAdapter(spec,device)
