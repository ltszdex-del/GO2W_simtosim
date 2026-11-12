"""Check student history, privileged-state exclusion and original motor context."""
import hashlib,json,sys,tempfile
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import bootstrap
from run import create_runtime
from handstand_student_adapter import HandstandStudentAdapter

env,controller=create_runtime()
teacher=controller.adapters['handstand']
assert teacher.ck['kind']=='teacher','Run this contract check before replacing the teacher'
spec=dict(teacher.spec)
with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
    path=Path(folder)/'synthetic_student.pt'
    model=teacher.networks.Student().eval()
    torch.save(dict(kind='student',interface=teacher.saved,model=model.state_dict()),path)
    spec.update(checkpoint=str(path),teacher_checkpoint=teacher.path,
        teacher_sha256=hashlib.sha256(Path(teacher.path).read_bytes()).hexdigest())
    student=HandstandStudentAdapter(spec,'cpu');student.load();student.bind_env(env)
    student.reset(env.snapshot())
    with torch.inference_mode():
        obs=student.observe(env.snapshot())
        assert obs.shape==(1,2657)
        history=obs[:,:2600].reshape(1,50,52)
        assert torch.equal(history,history[:,:1].expand_as(history))
        before=student.infer(obs)
        original_blocks=student.ctx.blocks
        def poisoned_blocks(*args,**kwargs):
            proprio,goal,privileged=original_blocks(*args,**kwargs)
            return proprio,goal,torch.full_like(privileged,float('nan'))
        student.ctx.blocks=poisoned_blocks
        after_obs=student.observe(env.snapshot());after=student.infer(after_obs)
        assert torch.isfinite(after_obs).all() and torch.isfinite(after).all()
        assert torch.equal(obs,after_obs) and torch.equal(before,after)
        assert student.ctx.local_motor_subdivisions==5
        assert torch.equal(student.ctx.local_entry_heading,student.ctx.stable_heading())
print('PASS: 2657D student; 50-frame initial history; privileged NaNs excluded; local H fixes retained')
