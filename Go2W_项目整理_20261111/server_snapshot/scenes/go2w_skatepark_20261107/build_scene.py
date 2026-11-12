"""Self-contained, meter-scale Z-up static training terrain. Requires pxr + numpy."""
from pathlib import Path
import json,math,argparse
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux,Gf,Sdf

OUT=Path(__file__).resolve().parent
stage=Usd.Stage.CreateNew(str(OUT/'go2w_skatepark.usda'))
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Scope.Define(stage,'/World/Terrain');UsdGeom.Scope.Define(stage,'/World/Looks')
stage.SetMetadata('documentation','Go2W skatepark training terrain; meters, Z-up; static concave mesh collisions; robot not included.')
stage.GetRootLayer().customLayerData={'terrainVersion':'1.0','centralClearArea':'8x8m','units':'meters'}
palette={'concrete':(.43,.48,.51),'light':(.65,.69,.70),'blue':(.12,.43,.52),'orange':(.87,.43,.18),'green':(.28,.48,.39),'paint':(.83,.85,.75),'dark':(.20,.25,.28)}
materials={};manifest={'bounds_m':[-18,18,-14,14],'central_clear_area_m':[-4,4,-4,4],'obstacles':[],'units':'m','up_axis':'Z','friction':{'static':.9,'dynamic':.8,'restitution':0},'spawn_reference':'root z assumes standing Go2W approximately .42m; adjust to robot asset'}
render_meshes=[];colliders=[]
physicsmat=UsdShade.Material.Define(stage,'/World/Looks/ContactMaterial')
phys=UsdPhysics.MaterialAPI.Apply(physicsmat.GetPrim());phys.CreateStaticFrictionAttr(.9);phys.CreateDynamicFrictionAttr(.8);phys.CreateRestitutionAttr(0.)
for name,color in palette.items():
 mat=UsdShade.Material.Define(stage,'/World/Looks/'+name);shader=UsdShade.Shader.Define(stage,mat.GetPath().AppendChild('Surface'))
 shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.88)
 mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');materials[name]=mat

def mesh(path,verts,faces,color='concrete',collision=True):
 v=np.asarray(verts,dtype=float);m=UsdGeom.Mesh.Define(stage,path)
 m.CreatePointsAttr([Gf.Vec3f(*p) for p in v]);m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr([i for f in faces for i in f]);m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(False)
 m.CreateExtentAttr([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))]);UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(materials[color])
 if collision:
  UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
  UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
  UsdShade.MaterialBindingAPI(m.GetPrim()).Bind(physicsmat,materialPurpose='physics');colliders.append(path)
 render_meshes.append((v,faces,palette[color]));return m

BOX=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
def box(path,center,size,color='concrete',collision=True):
 x,y,z=np.array(size)/2;c=np.array(center)
 v=np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])+c
 return mesh(path,v,BOX,color,collision)

def transform(v,origin,yaw):
 a=math.radians(yaw);r=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
 return np.asarray(v)@r.T+np.array([*origin,0])

def profile(name,points,width,origin,yaw=0,color='concrete',kind='bank',details=None):
 # Counterclockwise polygon in local X/Z: bottom -> end -> reverse top.
 poly=[(points[0][0],-.08),(points[-1][0],-.08)]+list(reversed(points));k=len(poly)
 v=[(x,-width/2,z) for x,z in poly]+[(x,width/2,z) for x,z in poly]
 # End polygons can be nonconvex; triangulate the X/Z cross-section using ears.
 def ears(poly):
  ids=list(range(len(poly)));tri=[];p=np.array(poly)
  def cross(a,b,c):
   u,v=b-a,c-a
   return u[0]*v[1]-u[1]*v[0]
  while len(ids)>3:
   for j,i in enumerate(ids):
    a,b,c=ids[j-1],i,ids[(j+1)%len(ids)]
    if cross(p[a],p[b],p[c])<=1e-10:continue
    if any(cross(p[a],p[b],p[t])>=-1e-10 and cross(p[b],p[c],p[t])>=-1e-10 and cross(p[c],p[a],p[t])>=-1e-10 for t in ids if t not in (a,b,c)):continue
    tri.append([a,b,c]);ids.pop(j);break
   else:raise ValueError('Profile triangulation failed '+name)
  return tri+[ids]
 triangles=ears(poly);faces=triangles+[[c+k,b+k,a+k] for a,b,c in triangles]
 faces += [[i,i+k,(i+1)%k+k,(i+1)%k] for i in range(k)]
 mesh('/World/Terrain/'+name,transform(v,origin,yaw),faces,color)
 length=points[-1][0]-points[0][0];start=transform([[-1.0,0,.42]],origin,yaw)[0].tolist()
 end=transform([[length+1.,0,.42]],origin,yaw)[0].tolist()
 item=dict(name=name,type=kind,width_m=width,length_m=length,height_m=max(z for x,z in points),origin_xy_m=origin,yaw_deg=yaw,approach_root_pose_xyz=start,exit_floor_root_pose_xyz=end,profile_xz_m=points)
 if details:item.update(details)
 manifest['obstacles'].append(item)

def stairs(name,heights,treads,origin,yaw=90,width=2.2,color='blue',kind='stairs'):
 points=[(0,0)];x=0.
 for h,t in zip(heights,treads):
  points.append((x,h));x+=t;points.append((x,h))
 # Include level landing before the rear drop; approach is from the first step.
 profile(name,points,width,origin,yaw,color,kind,dict(tread_heights_m=heights,tread_depths_m=treads,landing_rear_drop=True))

box('/World/Terrain/Ground',(0,0,-.15),(36,28,.3),'dark')
# Progressive stairs: cumulative top heights, including nonuniform and wide steps.
stairs('S01_MicroSteps',[.03,.06,.09,.12,.15,.18],[.35]*5+[.8],[-14,6.5],color='green')
stairs('S02_LowSteps',[.06,.12,.18,.24,.30],[.40]*4+[.9],[-8.5,6.5])
stairs('S03_StandardSteps',[.10,.20,.30,.40,.50],[.40]*4+[1.0],[-3,6.5])
stairs('S04_HighSteps',[.15,.30,.45,.60],[.45]*3+[1.1],[3,6.5],color='orange')
stairs('S05_IrregularSteps',[.04,.12,.18,.30,.38],[.28,.48,.32,.58,.9],[8.5,6.5],color='orange',kind='irregular_stairs')
stairs('S06_DeepTerraces',[.18,.36,.54],[.9,.9,1.2],[14,6.5],color='light',kind='deep_terraces')
# A staircase that also descends, suitable for traversal in either direction.
stairs('S07_UpAndDown',[.08,.16,.24,.32,.24,.16,.08],[.42,.42,.42,.8,.42,.42,.42],[-8,-4],yaw=90,width=2.4,color='green',kind='up_down_stairs')
# South banks and a tabletop, plus a low broad grind/manual pad.
profile('R01_GentleBank',[(0,0),(2.5,.25),(3.5,.25),(6,0)],2.6,[-16,-10],color='green',details={'slope_deg':math.degrees(math.atan(.1))})
profile('R02_Tabletop',[(0,0),(1.8,.5),(3.2,.5),(5,0)],3.,[-7,-10],color='light')
profile('R03_SteepBank',[(0,0),(1.5,.6),(2.5,.6),(4,0)],2.6,[2,-10],color='orange')
profile('L01_ManualPad',[(0,0),(0,.12),(3,.12),(3,0)],1.5,[-15,3],color='blue',kind='low_ledge')
# Quarter-pipe rises tangentially from flat ground; polygon segments <= about 2cm.
radius=1.0;theta=np.linspace(0,math.pi/2,97)
curve=[(float(radius*math.sin(t)),float(radius*(1-math.cos(t)))) for t in theta]+[(1.6,1.)]
profile('Q01_QuarterPipe',curve,3.,[10,-10],color='blue',kind='quarter_pipe',details={'radius_m':1.,'arc_segments':96})
# Smooth shallow halfpipe: quarter-circle walls, 3m flat middle, 3m width.
radius=.8;t=np.linspace(0,math.pi/2,81)
left=[(float(radius*(1-math.cos(a))),float(radius*(1-math.sin(a)))) for a in t]
right=[(float(radius+3+radius*math.sin(a)),float(radius*(1-math.cos(a)))) for a in t[1:]]
profile('Q02_MiniHalfPipe',left+[(radius+3,0)]+right,3.4,[12,-4],yaw=90,color='light',kind='half_pipe',details={'radius_m':.8,'flat_length_m':3.})
# Four-sided bank pyramid; 5m square footprint, 1.6m square top.
v=[[-2.5,-2.5,0],[2.5,-2.5,0],[2.5,2.5,0],[-2.5,2.5,0],[-.8,-.8,.45],[.8,-.8,.45],[.8,.8,.45],[-.8,.8,.45]]
mesh('/World/Terrain/R04_FourWayPyramid',transform(v,[-12,-2],0),BOX,'concrete')
manifest['obstacles'].append(dict(name='R04_FourWayPyramid',type='four_way_bank',origin_xy_m=[-12,-2],width_m=5.,length_m=5.,height_m=.45,approach_root_pose_xyz=[-15.5,-2,.42]))
# Thin painted markings have no collision. Center remains obstacle-free.
for i,(cx,cy,sx,sy) in enumerate([(0,-4,8,.04),(0,4,8,.04),(-4,0,.04,8),(4,0,.04,8)]):
 box('/World/Markings/CenterBoundary'+str(i),(cx,cy,.001),(sx,sy,.001),'paint',False)
for i,x in enumerate([-14,-8.5,-3,3,8.5,14]):
 box('/World/Markings/StairApproach'+str(i),(x,5.7,.001),(1.7,.04,.001),'paint',False)
# Low border outside the usable field; no thin rails in the traversal zone.
for i,(c,s) in enumerate([((0,-14.1,.10),(36.4,.2,.2)),((0,14.1,.10),(36.4,.2,.2)),((-18.1,0,.10),(.2,28,.2)),((18.1,0,.10),(.2,28,.2))]):box('/World/Terrain/Border'+str(i),c,s,'concrete')
manifest['spawn_zones']=[dict(name='center_flat',xy_bounds_m=[-3,3,-3,3],standing_root_z_m=.42),dict(name='north_approaches',xy_bounds_m=[-16,16,4.6,5.6],standing_root_z_m=.42)]
manifest['collider_count']=len(colliders)
stage.GetRootLayer().Save();stage.GetRootLayer().Export(str(OUT/'go2w_skatepark.usd'))
# Separate preview wrapper; training asset has no physics scene, lights or camera.
preview=Usd.Stage.CreateNew(str(OUT/'preview.usda'));UsdGeom.SetStageUpAxis(preview,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(preview,1.)
pr=UsdGeom.Xform.Define(preview,'/World');preview.SetDefaultPrim(pr.GetPrim());UsdGeom.Xform.Define(preview,'/World/Park').GetPrim().GetReferences().AddReference('./go2w_skatepark.usd')
scene=UsdPhysics.Scene.Define(preview,'/World/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
light=UsdLux.DomeLight.Define(preview,'/World/Sky');light.CreateIntensityAttr(900.)
sun=UsdLux.DistantLight.Define(preview,'/World/Sun');sun.CreateIntensityAttr(2500.);UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,20))
camera=UsdGeom.Camera.Define(preview,'/World/Camera');matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(29,-36,31),Gf.Vec3d(0,0,0),Gf.Vec3d(0,0,1)).GetInverse();UsdGeom.Xformable(camera).AddTransformOp().Set(matrix);camera.CreateClippingRangeAttr(Gf.Vec2f(.1,500));camera.CreateFocalLengthAttr(21)
preview.GetRootLayer().Save()
(OUT/'terrain_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'preview_geometry.npz',vertices=np.array([m[0] for m in render_meshes],dtype=object),faces=np.array([m[1] for m in render_meshes],dtype=object),colors=np.array([m[2] for m in render_meshes]))
print('BUILT',len(manifest['obstacles']),'obstacles',len(colliders),'static colliders')
