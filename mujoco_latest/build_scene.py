"""Convert the deployed URDF and all 19 static USD terrain colliders."""
import bootstrap
import copy
import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import mujoco
import numpy as np

ROOT = bootstrap.ROOT

def build(config):
    out = ROOT / 'generated'; out.mkdir(exist_ok=True)
    # Reuse the previously audited inertia-preserving URDF importer. Its visual
    # lookup is redirected to the installed official meshes; collision/inertia
    # input always remains the CURRENT deployed URDF.
    source = bootstrap.WORKSPACE / 'audit_sim2sim_20261104/staged/import_training_urdf.py'
    namespace = {'__file__': str(source), '__name__': 'audited_urdf_import'}
    code = source.read_text(encoding='utf-8').replace(
        'visual_path = urdf_path.parent.parent / "go2w.xml"',
        'visual_path = Path(' + repr(str(bootstrap.WORKSPACE / '.tmp_unitree_mujoco/unitree_robots/go2w/go2w.xml')) + ')')
    code = code.replace('base_inertia = root_inertia(urdf)', '''base_inertia = root_inertia(urdf)
    for mesh in urdf.iter('mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith('package://go2w_description/'):
            mesh.set('filename', str(urdf_path.parent.parent / filename.split('package://go2w_description/', 1)[1]))''')
    exec(compile(code, str(source), 'exec'), namespace)
    robot = out / 'robot.xml'
    import shutil
    local_asset=out/'source';local_asset.mkdir(exist_ok=True)
    shutil.copytree(Path(config['robot_asset_root'])/'meshes',local_asset/'meshes',dirs_exist_ok=True)
    (local_asset/'urdf').mkdir(exist_ok=True)
    local_urdf=local_asset/'urdf/go2w_description.urdf'
    shutil.copy2(Path(config['robot_asset_root'])/'urdf/go2w_description.urdf',local_urdf)
    namespace['build'](local_urdf, robot)
    tree = ET.parse(robot); root = tree.getroot()
    option = root.find('option'); option.set('integrator', 'implicitfast')
    option.set('iterations', '100'); option.set('tolerance', '1e-10')
    for geom in root.iter('geom'):
        if geom.get('group') == '3':
            geom.set('condim', '3'); geom.set('friction', '.8 0 0')
            geom.set('solref', '.01 1'); geom.set('solimp', '.95 .99 .001')
    # Velocity actuators implement the wheel drive implicitly. Joint total
    # actuator force range caps both explicit torque and velocity drive together.
    actuator = root.find('actuator')
    for side in ('FR','FL','RR','RL'):
        name = side + '_wheel_joint'
        joint = next(j for j in root.iter('joint') if j.get('name') == name)
        joint.set('actuatorfrcrange', '-23.5 23.5')
        ET.SubElement(actuator, 'velocity', name=side+'_drive', joint=name, kv='100',
                      ctrlrange='-30.1 30.1', forcerange='-23.5 23.5')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1', directional='true', diffuse='.8 .8 .8')
    assets = root.find('asset')
    ET.SubElement(assets,'texture',name='checker',type='2d',builtin='checker',rgb1='.22 .26 .3',rgb2='.32 .36 .4',width='256',height='256')
    ET.SubElement(assets,'material',name='groundmat',texture='checker',texrepeat='18 14',reflectance='.1')
    colliders = []
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage = Usd.Stage.Open(config['terrain']['usd_path'])
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        if not prim.IsA(UsdGeom.Mesh): raise ValueError('Unsupported collider: '+str(prim.GetPath()))
        mesh = UsdGeom.Mesh(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = np.array([transform.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
        counts = list(mesh.GetFaceVertexCountsAttr().Get()); indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces=[]; offset=0
        for n in counts:
            face=indices[offset:offset+n];offset+=n
            for i in range(1,n-1): faces.extend([face[0],face[i],face[i+1]])
        name = 'terrain_' + str(len(colliders))
        # MuJoCo mesh collision is convex. This terrain is split into 19 convex
        # pieces; verify each piece rather than silently convexify a concave ramp.
        from scipy.spatial import ConvexHull
        hull=ConvexHull(points)
        signed=[]
        for i in range(0,len(faces),3):
            a,b,c=points[faces[i:i+3]]
            signed.append(np.dot(a,np.cross(b,c))/6)
        mesh_volume=abs(sum(signed))
        pieces=[points]
        if abs(mesh_volume-hull.volume)>max(1e-5,hull.volume*1e-4):
            # All nonconvex source pieces are vertical extrusions of a monotone
            # staircase/ramp profile. Split each upper profile edge into an
            # exact convex trapezoidal prism, preserving the original surface.
            n=len(points)//2; profile=points[:n];shift=points[n:]-profile
            if not np.allclose(shift,shift[0],atol=1e-5): raise ValueError('Non-extruded terrain')
            axis=0 if np.ptp(profile[:,0])>np.ptp(profile[:,1]) else 1
            bottom=points[:,2].min();pieces=[]
            for i,a in enumerate(profile):
                b=profile[(i+1)%n]
                if abs(a[axis]-b[axis])<1e-7 or max(a[2],b[2])<bottom+1e-7:continue
                lowa=a.copy();lowb=b.copy();lowa[2]=lowb[2]=bottom
                side=np.array([lowa,lowb,b,a]);pieces.append(np.r_[side,side+shift[0]])
            volume=sum(ConvexHull(piece).volume for piece in pieces)
            if abs(volume-mesh_volume)>max(1e-5,mesh_volume*1e-4):raise ValueError('Terrain decomposition volume mismatch')
        for k,piece in enumerate(pieces):
            key=name+'_'+str(k)
            ET.SubElement(assets,'mesh',name=key,vertex=' '.join(map(str,piece.flatten())))
            ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,friction='.8 0 0',condim='3',material='groundmat',solref='.01 1')
        colliders.append(dict(path=str(prim.GetPath()),vertices=len(points),triangles=len(faces)//3,volume=mesh_volume,convex_pieces=len(pieces)))
    if len(colliders)!=19: raise ValueError(f'Expected 19 colliders, got {len(colliders)}')
    path=out/'scene.xml';tree.write(path,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path))
    report=dict(mass_kg=float(model.body_mass.sum()),nq=model.nq,nv=model.nv,nu=model.nu,
                terrain=colliders,urdf=str(Path(config['robot_asset_root'])/'urdf/go2w_description.urdf'),
                physics_dt=model.opt.timestep,integrator='implicitfast')
    (out/'model_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return path

if __name__=='__main__': print(build(bootstrap.load_config()))
