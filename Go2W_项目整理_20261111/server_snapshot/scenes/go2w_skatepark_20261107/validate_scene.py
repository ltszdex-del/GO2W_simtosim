"""Check units, standalone references, mesh topology and collision metadata."""
from pathlib import Path
import json,collections
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdUtils
r=Path(__file__).resolve().parent;s=Usd.Stage.Open(str(r/'go2w_skatepark.usd'))
assert s and s.GetDefaultPrim() and UsdGeom.GetStageUpAxis(s)=='Z' and UsdGeom.GetStageMetersPerUnit(s)==1.
count=0;triangles=0
for p in s.Traverse():
 if not p.IsA(UsdGeom.Mesh):continue
 m=UsdGeom.Mesh(p);v=np.array(m.GetPointsAttr().Get());n=list(m.GetFaceVertexCountsAttr().Get());ids=list(m.GetFaceVertexIndicesAttr().Get())
 assert np.isfinite(v).all() and sum(n)==len(ids) and min(ids)>=0 and max(ids)<len(v),p.GetPath()
 edges=collections.Counter();directed=collections.Counter();at=0
 for size in n:
  face=ids[at:at+size];at+=size
  for a,b in zip(face,face[1:]+face[:1]):edges[tuple(sorted([a,b]))]+=1;directed[(a,b)]+=1
  area=sum((np.cross(v[face[i]]-v[face[0]],v[face[i+1]]-v[face[0]]) for i in range(1,size-1)),np.zeros(3))
  assert np.linalg.norm(area)>1e-9,('degenerate face',p.GetPath(),face)
  triangles+=size-2
 assert all(k==2 for k in edges.values()),('nonmanifold',p.GetPath())
 assert all(directed[(a,b)]==directed[(b,a)] for a,b in edges),('winding',p.GetPath())
 if p.HasAPI(UsdPhysics.CollisionAPI):
  count+=1;assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()=='none';assert not p.HasAPI(UsdPhysics.RigidBodyAPI)
  mat,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics');assert mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
  if str(p.GetPath()).startswith('/World/Terrain/') and p.GetName()!='Ground' and not p.GetName().startswith('Border'):
   assert not ((v[:,0].min()<4 and v[:,0].max()>-4) and (v[:,1].min()<4 and v[:,1].max()>-4)),('central obstacle',p.GetPath())
assert count==json.loads((r/'terrain_manifest.json').read_text())['collider_count']
layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(r/'go2w_skatepark.usd'));assert not unresolved
assert len(layers)==1 and not assets,('external dependency',layers,assets)
result=dict(status='passed',colliders=count,triangles=triangles,checks=['meters/Z-up/defaultPrim','finite points and indices','watertight consistently wound nondegenerate meshes','static triangle collisions','physics material bindings','central 8x8m clearance','no external assets'],limitations='USD structure verified; robot contact and traversal require later task integration')
(r/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
