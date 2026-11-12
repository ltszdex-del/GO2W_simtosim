from pathlib import Path
import sys,hashlib,torch,json
r=Path(sys.argv[1]);parent=Path(sys.argv[2]);sys.path.insert(0,str(r))
from config import contract,URDF,PPO
from learning import Policy
ck=torch.load(parent,map_location='cpu',weights_only=False)
old=ck['manifest'];old_root=parent.parents[2]
for name,digest in old['source_sha256'].items():
    assert hashlib.sha256((old_root/name).read_bytes()).hexdigest()==digest,name
model=Policy();model.load_state_dict(ck['model'],strict=True)
manifest=contract();manifest['curriculum']=old['curriculum']
manifest['urdf_sha256']=hashlib.sha256(URDF.read_bytes()).hexdigest()
manifest['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')}
ck['manifest']=manifest
for group in ck['optimizer']['param_groups']:group['lr']=PPO['lr']
(r/'resume').mkdir(exist_ok=True);torch.save(ck,r/'resume/parent.pt')
(r/'MIGRATION.json').write_text(json.dumps(dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    preserved_optimizer=True,iterations=int(sys.argv[3]),goal_iteration=ck['iteration']+int(sys.argv[3]),old_manifest=old,new_manifest=manifest),indent=2))
