import xml.etree.ElementTree as ET
import torch,math
def stance_feet(urdf,names,q):
    tree=ET.parse(urdf).getroot();angles=dict(zip(names,q));children={j.find('child').get('link') for j in tree.findall('joint')}
    root=next(x.get('name') for x in tree.findall('link') if x.get('name') not in children)
    poses={root:torch.eye(4,dtype=torch.float64)};todo=list(tree.findall('joint'))
    def numbers(x,default='0 0 0'):return torch.tensor([float(v) for v in (x or default).split()],dtype=torch.float64)
    while todo:
        changed=False
        for j in list(todo):
            parent=j.find('parent').get('link')
            if parent not in poses:continue
            o=j.find('origin');xyz=numbers(None if o is None else o.get('xyz'));rpy=numbers(None if o is None else o.get('rpy'))
            rx,ry,rz=[float(v) for v in rpy];cx,sx=math.cos(rx),math.sin(rx);cy,sy=math.cos(ry),math.sin(ry);cz,sz=math.cos(rz),math.sin(rz)
            R=torch.tensor([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]],dtype=torch.float64)
            axis=j.find('axis');a=numbers('1 0 0' if axis is None else axis.get('xyz'));a=a/a.norm();x,y,z=a
            K=torch.tensor([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=torch.float64);angle=angles.get(j.get('name'),0.)
            rot=torch.eye(3,dtype=torch.float64)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
            T=torch.eye(4,dtype=torch.float64);T[:3,:3]=R@rot;T[:3,3]=xyz
            poses[j.find('child').get('link')]=poses[parent]@T;todo.remove(j);changed=True
        assert changed,'URDF tree unresolved'
    offsets=collision_offsets(urdf)
    return torch.stack([(poses[n]@torch.tensor([*offsets[i],1.],dtype=torch.float64))[:3] for i,n in enumerate(('FR_foot','FL_foot','RR_foot','RL_foot'))]).float().tolist()
def collision_offsets(urdf):
    t=ET.parse(urdf).getroot();links={x.get('name'):x for x in t.findall('link')}
    return [[float(v) for v in links[n].find('collision/origin').get('xyz','0 0 0').split()] for n in ('FR_foot','FL_foot','RR_foot','RL_foot')]
if __name__=='__main__':
    from pathlib import Path
    p=Path('/home/ubuntu/Go2W/go2w_wheel_drive_20261107/assets/go2w_description/urdf/go2w_description.urdf');t=ET.parse(p).getroot()
    for n in ('FR_foot','FR_foot_joint'):
        node=next(x for x in t if x.get('name')==n);print(ET.tostring(node,encoding='unicode'))
