"""USD arrows: green reference, blue measured velocity. 1m length = 1m/s."""
import math
from pxr import UsdGeom,Gf
import omni.usd

class VelocityArrows:
    def __init__(self):
        self.stage=omni.usd.get_context().get_stage()
        self.roots=[]
        for name,color in [('Reference',(0.,1.,0.)),('Actual',(0.,.35,1.))]:
            path='/World/Velocity'+name
            root=UsdGeom.Xform.Define(self.stage,path)
            shaft=UsdGeom.Cylinder.Define(self.stage,path+'/Shaft');shaft.CreateAxisAttr('Z');shaft.CreateRadiusAttr(.012)
            tip=UsdGeom.Cone.Define(self.stage,path+'/Tip');tip.CreateAxisAttr('Z');tip.CreateRadiusAttr(.04)
            shaft.CreateDisplayColorAttr([color]);tip.CreateDisplayColorAttr([color])
            shaft_pos=UsdGeom.Xformable(shaft).AddTranslateOp()
            tip_pos=UsdGeom.Xformable(tip).AddTranslateOp()
            self.roots.append((root,shaft,tip,root.MakeMatrixXform(),shaft_pos,tip_pos))
    def update(self,env):
        p=env._physical;yaw=float(p['yaw'][0]);co,si=math.cos(yaw),math.sin(yaw)
        ref=env.command.velocity[0,:2].cpu().tolist()
        actual=env.robot.data.root_lin_vel_w.torch[0,:2].cpu().tolist()
        ref=[co*ref[0]-si*ref[1],si*ref[0]+co*ref[1]]
        pos=env.robot.data.root_pos_w.torch[0,:3].cpu().tolist()
        for i,((root,shaft,tip,matrix,shaft_pos,tip_pos),v) in enumerate(zip(self.roots,[ref,actual])):
            length=math.hypot(*v)
            imageable=UsdGeom.Imageable(root.GetPrim())
            if length<.01:imageable.MakeInvisible();continue
            imageable.MakeVisible()
            direction=Gf.Vec3d(v[0]/length,v[1]/length,0.)
            transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),direction))
            transform.SetTranslateOnly(Gf.Vec3d(pos[0],pos[1],pos[2]+.5+i*.15))
            matrix.Set(transform)
            head=min(.12,length*.3);body=length-head
            shaft.CreateHeightAttr(body);tip.CreateHeightAttr(head)
            shaft_pos.Set(Gf.Vec3d(0,0,body/2))
            tip_pos.Set(Gf.Vec3d(0,0,body+head/2))
