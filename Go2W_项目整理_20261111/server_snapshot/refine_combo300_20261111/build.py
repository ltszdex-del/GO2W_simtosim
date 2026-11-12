from pathlib import Path
r=Path(__file__).parent
def edit(kind,name,old,new):
    p=r/kind/name;s=p.read_text();assert old in s,(kind,name,old);p.write_text(s.replace(old,new))
edit('wheel','training_schedule.py',"self.device=device;self.group=torch.arange(n,device=device)%16", "self.device=device\n        lookup=torch.tensor([0,1,4,5]+list(range(8,16))*2,device=device)\n        self.group=lookup[torch.arange(n,device=device)%20]")
edit('step','task_schedule.py','lookup=torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,7,8,9,7,8],device=device)', 'lookup=torch.tensor([0,1,5,6]+[3,4]*6+[7,8,9]*8,device=device)')
edit('step','task_schedule.py','(torch.arange(n,device=device)//2)%20','(torch.arange(n,device=device)//2)%40')
# Evaluation must use the same 3 m/s forward cap as the training contract.
edit('wheel','env.py',"getattr(self,'drive_speed_cap',2.5) if self.cfg.training else 2.5", "getattr(self,'drive_speed_cap',COMMAND['speed_limits'][0]) if self.cfg.training else COMMAND['speed_limits'][0]")
for name in ('run_one.py','launch.py','training_snapshot.py','stop_stream.py'):
    p=r/name;s=p.read_text().replace('refine_followup_20261111','refine_combo300_20261111').replace('go2w_{kind}_followup_20261111','go2w_{kind}_combo300_20261111')
    s=s.replace("updates=400 if kind=='wheel' else 200",'updates=300').replace("goal=2399 if kind=='wheel' else 9599","goal=2699 if kind=='wheel' else 9899").replace("[('wheel',1999),('step',9399)]","[('wheel',2399),('step',9599)]")
    p.write_text(s)
for kind in ('wheel','step'):
    with (r/kind/'config.py').open('a') as f:f.write("\nGAIT['sampling_revision']='combo_focused_300_20261111'\n")
