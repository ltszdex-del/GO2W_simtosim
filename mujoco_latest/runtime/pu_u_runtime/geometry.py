import json,torch
from pathlib import Path
from isaaclab.utils.math import quat_apply
DATA=json.loads((Path(__file__).resolve().parent/'collision_geometry.json').read_text())
def clearances(env):
 if not hasattr(env,'hulls'):
  env.hulls={name:torch.tensor(points,device=env.device,dtype=torch.float32) for name,points in DATA.items()}
 d=env.robot.data;result={}
 for name,v in env.hulls.items():
  points=quat_apply(d.root_quat_w.torch[:,None].expand(-1,len(v),-1),v[None].expand(env.num_envs,-1,-1))
  result[name]=points[:,:,2].amin(-1)+d.root_pos_w.torch[:,2]-env.scene.env_origins[:,2]
 return result['base'],torch.minimum(result['Head_upper'],result['Head_lower'])

def roll_pitch(env):
 g=env.robot.data.projected_gravity_b.torch
 return torch.atan2(-g[:,1],-g[:,2]),g[:,0].clamp(-1,1).asin()

def limb_clearances(env):
 if not hasattr(env,'limb_hulls'):
  data=json.loads((Path(__file__).resolve().parent/'limb_collision_geometry.json').read_text())
  env.limb_hulls=[(name,env.robot.find_bodies(name,preserve_order=True)[0][0],torch.tensor(v,device=env.device)) for name,v in data.items()]
 result={leg:[] for leg in ('FR','FL','RR','RL')};d=env.robot.data
 for name,index,v in env.limb_hulls:
  points=quat_apply(d.body_link_quat_w.torch[:,index,None,:].expand(-1,len(v),-1),v[None].expand(env.num_envs,-1,-1))
  gap=points[:,:,2].amin(-1)+d.body_link_pos_w.torch[:,index,2]-env.scene.env_origins[:,2]
  result[name[:2]].append(gap)
 return torch.stack([torch.stack(result[leg],-1).amin(-1) for leg in ('FR','FL','RR','RL')],-1)
