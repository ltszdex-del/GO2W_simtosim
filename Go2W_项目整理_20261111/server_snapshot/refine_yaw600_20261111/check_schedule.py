from pathlib import Path
import sys,torch,json
kind=sys.argv[1];root=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/home/ubuntu/Go2W')/f'go2w_{kind}_yaw600_20261111';sys.path.insert(0,str(root))
if kind=='wheel':
    from training_schedule import TrainingSchedule
    n=2048;s=TrainingSchedule(n,'cpu',42)
    proportions=dict(combo=float(s.combo_mask.float().mean()),pure=float(((s.group>=4)&(s.group<8)).float().mean()),straight=float((s.group<4).float().mean()))
    assert .79<proportions['combo']<.81
else:
    from task_schedule import TrainingSchedule
    n=2048;s=TrainingSchedule(n,'cpu',42)
    proportions=dict(move_yaw=float((s.group>=7).float().mean()),diagonal=float(((s.group==3)|(s.group==4)).float().mean()),pure=float(s.qe_mask.float().mean()))
    assert .58<proportions['move_yaw']<.62 and .29<proportions['diagonal']<.32
active=torch.zeros(n,dtype=torch.bool);moving=torch.zeros(n,dtype=torch.bool)
for k in range(550):
    a=s.tick(3,torch.full((n,),k*.02),.02)
    assert torch.isfinite(a).all() and (a.abs()<=1).all()
    moving|=a[:,:2].abs().sum(-1)>0
    active|=(a[:,:2].abs().sum(-1)>0)&(a[:,2].abs()>0)
    if kind=='step':
        assert torch.equal(a[0::2,0],a[1::2,0]) and torch.equal(a[0::2,1:],-a[1::2,1:])
    if k==540:assert not a.any(),'End-of-episode stop missing'
assert float(active.float().mean())>.58
result=dict(kind=kind,environment_fractions=proportions,ever_combined_fraction=float(active.float().mean()),status='passed')
(root/'schedule_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
