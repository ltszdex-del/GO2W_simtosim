"""Load the test terrain without changing any policy or robot parameters."""
import hashlib
from pathlib import Path

DEFAULT_USD='/home/ubuntu/Go2W/scenes/go2w_skatepark_20261107/go2w_skatepark.usd'

def apply_terrain(ec,config,override=None):
    from pxr import Usd,UsdGeom,UsdPhysics
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils
    spec=config.get('terrain',{})
    kind=override or spec.get('kind','flat')
    if kind=='flat':
        return dict(kind='flat',prim_path=ec.scene.ground.prim_path,spawn_xy=[0.,0.])
    if kind!='skatepark':raise ValueError(f'Unknown terrain {kind!r}')
    usd=Path(spec.get('usd_path',DEFAULT_USD)).resolve(strict=True)
    stage=Usd.Stage.Open(str(usd))
    if not stage or not stage.GetDefaultPrim():raise ValueError(f'Invalid terrain USD: {usd}')
    if UsdGeom.GetStageUpAxis(stage)!='Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9:
        raise ValueError('Test terrain must use meters and Z-up')
    colliders=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):raise ValueError('Terrain must not introduce a second PhysicsScene')
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Terrain must contain only static geometry')
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.IsA(UsdGeom.Mesh) and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='none':
                raise ValueError(f'Terrain mesh needs triangle collision: {prim.GetPath()}')
            colliders.append(str(prim.GetPath()))
    if len(colliders)!=19:raise ValueError(f'Expected 19 skatepark colliders, found {len(colliders)}')
    # Replace the existing asset. Do not spawn a second same-height floor.
    ec.scene.ground=AssetBaseCfg(prim_path='/World/TestTerrain',collision_group=-1,
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd)))
    if max(abs(float(x)) for x in ec.scene.robot.init_state.pos[:2])>3.:
        raise ValueError('Default skatepark reset must stay in the central clear area')
    return dict(kind=kind,usd_path=str(usd),sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        prim_path='/World/TestTerrain',collider_count=len(colliders),units='m',up_axis='Z',
        spawn_xy=list(ec.scene.robot.init_state.pos[:2]),central_clear_area_xy=[-4.,4.,-4.,4.],
        bounds_xy=[-18.,18.,-14.,14.],floor_height=0.,original_floor_replaced=True)

def inspect_loaded_terrain(report):
    """Validate the composed live stage, including material remapping on reference."""
    if report['kind']=='flat':return report
    import omni.usd
    from pxr import Usd,UsdPhysics,UsdShade
    stage=omni.usd.get_context().get_stage()
    root=stage.GetPrimAtPath(report['prim_path'])
    if not root:raise RuntimeError('Referenced test terrain is missing')
    meshes=[]
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise RuntimeError(f'Disabled terrain collision: {prim.GetPath()}')
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise RuntimeError(f'Unresolved terrain material: {prim.GetPath()}')
            meshes.append(str(prim.GetPath()))
    if len(meshes)!=report['collider_count']:raise RuntimeError(f'Terrain collider mismatch: {meshes}')
    if stage.GetPrimAtPath('/World/ground'):
        raise RuntimeError('Old ground plane overlaps the test terrain')
    report['loaded_collision_paths']=meshes
    report['live_stage_checked']=True
    return report
