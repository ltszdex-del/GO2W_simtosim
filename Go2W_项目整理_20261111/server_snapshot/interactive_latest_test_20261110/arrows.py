"""Green reference / blue actual: planar m/s and COM vertical rad/s arrows."""
import math
from pxr import UsdGeom,Gf
import omni.usd


def yaw_vector(rate):
    return (0.,0.,float(rate)*.30)


class VelocityArrows:
    def __init__(self):
        self.stage=omni.usd.get_context().get_stage()
        self.roots=[]
        for name,color in [('Reference',(0.,1.,0.)),('Actual',(0.,.35,1.)),
                           ('YawReference',(0.,1.,0.)),('YawActual',(0.,.35,1.))]:
            path='/World/Velocity'+name
            root=UsdGeom.Xform.Define(self.stage,path)
            shaft=UsdGeom.Cylinder.Define(self.stage,path+'/Shaft');shaft.CreateAxisAttr('Z')
            tip=UsdGeom.Cone.Define(self.stage,path+'/Tip');tip.CreateAxisAttr('Z')
            shaft.CreateRadiusAttr(.012);tip.CreateRadiusAttr(.04)
            shaft.CreateDisplayColorAttr([color]);tip.CreateDisplayColorAttr([color])
            # Coaxial angular arrows: a dashed green reference surrounds the
            # thinner continuous blue measurement, both begin exactly at COM.
            segments=[]
            if name=='YawReference':
                UsdGeom.Imageable(shaft).MakeInvisible()
                tip.CreateDisplayOpacityAttr([.45])
                tip.CreateRadiusAttr(.055)
                for i in range(12):
                    part=UsdGeom.Cylinder.Define(self.stage,path+f'/Dash{i}')
                    part.CreateAxisAttr('Z');part.CreateRadiusAttr(.022)
                    part.CreateDisplayColorAttr([color])
                    segments.append((part,UsdGeom.Xformable(part).AddTranslateOp()))
            self.roots.append((root,shaft,tip,root.MakeMatrixXform(),
                UsdGeom.Xformable(shaft).AddTranslateOp(),UsdGeom.Xformable(tip).AddTranslateOp(),segments))
        self.checked=False

    def update(self,env):
        yaw=float(env._physical['yaw'][0]);co,si=math.cos(yaw),math.sin(yaw)
        ref=env.command.velocity[0,:2].cpu().tolist()
        actual=env.robot.data.root_lin_vel_w.torch[0,:2].cpu().tolist()
        ref=[co*ref[0]-si*ref[1],si*ref[0]+co*ref[1],0.]
        actual=[*actual,0.]
        d=env.robot.data
        weights=d.body_mass.torch[0];weights=weights/weights.sum()
        com=(d.body_com_pos_w.torch[0]*weights[:,None]).sum(0).cpu().tolist()
        pos=d.root_pos_w.torch[0,:3].cpu().tolist()
        wz_ref=float(env.command.velocity[0,2]);wz_actual=float(d.root_ang_vel_w.torch[0,2])
        vectors=[ref,actual,yaw_vector(wz_ref),yaw_vector(wz_actual)]
        origins=[[pos[0],pos[1],pos[2]+.5],[pos[0],pos[1],pos[2]+.65],com,com]
        for i,((root,shaft,tip,matrix,shaft_pos,tip_pos,segments),v,origin) in enumerate(zip(self.roots,vectors,origins)):
            length=math.sqrt(sum(x*x for x in v))
            assert math.isfinite(length) and all(math.isfinite(x) for x in origin)
            imageable=UsdGeom.Imageable(root.GetPrim())
            if length<.003:imageable.MakeInvisible();continue
            imageable.MakeVisible()
            direction=Gf.Vec3d(*(x/length for x in v))
            transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),direction))
            transform.SetTranslateOnly(Gf.Vec3d(*origin));matrix.Set(transform)
            head=min(.12,length*.3);body=length-head
            shaft.CreateHeightAttr(body);tip.CreateHeightAttr(head)
            shaft_pos.Set(Gf.Vec3d(0,0,body/2));tip_pos.Set(Gf.Vec3d(0,0,body+head/2))
            for j,(part,translate) in enumerate(segments):
                part.CreateHeightAttr(body/12*.55)
                translate.Set(Gf.Vec3d(0,0,body/12*(j+.5)))
            if i>=2:
                assert (transform.ExtractTranslation()-Gf.Vec3d(*com)).GetLength()<1e-9
                assert (transform.TransformDir(Gf.Vec3d(0,0,length))-Gf.Vec3d(*v)).GetLength()<1e-8
        if not self.checked:
            assert yaw_vector(3.)==(0.,0.,.8999999999999999)
            assert yaw_vector(-3.)[2]<0
            print('[YAW_ARROWS_READY] true mass-weighted COM; green dashed=reference; blue=actual; +Z=CCW; 0.30m per rad/s',flush=True)
            self.checked=True
