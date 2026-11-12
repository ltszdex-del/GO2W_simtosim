"""Shared training/deployment command integrator. SI units, yaw-only body frame."""
import math
from directional_reference import advance_axis
from yaw_envelope import yaw_limit,brake_to_limit

MOTION_KEYS=set('WASDQE')
ALL_KEYS=MOTION_KEYS|set('HPURMK')
PROFILE_KEYS=('min_forward_m_s','max_forward_m_s','max_lateral_m_s','max_yaw_rad_s',
    'forward_accel_m_s2','backward_accel_m_s2','lateral_accel_m_s2',
    'translation_accel_limit_m_s2','forward_brake_m_s2','lateral_brake_m_s2',
    'yaw_accel_rad_s2','yaw_brake_rad_s2','wheel_reference_speed_m_s','wheel_envelope_lever',
    'yaw_cruise_limit_rad_s','yaw_speed_decay')

def profile_from_contract(command,control):
    a=command['axis_acceleration'];b=command['axis_brake'];limits=command['speed_limits']
    back=command.get('backward_acceleration',a[0])
    return dict(min_forward_m_s=command['min_forward_speed'],max_forward_m_s=limits[0],
        max_lateral_m_s=limits[1],max_yaw_rad_s=command['yaw_speed_limit'],
        yaw_cruise_limit_rad_s=command.get('yaw_cruise_limit',command['yaw_speed_limit']),
        yaw_speed_decay=command.get('yaw_speed_decay',3.),
        forward_accel_m_s2=a[0],backward_accel_m_s2=back,lateral_accel_m_s2=a[1],
        translation_accel_limit_m_s2=math.hypot(max(a[0],back),a[1]),
        forward_brake_m_s2=b[0],lateral_brake_m_s2=b[1],yaw_accel_rad_s2=command['yaw_acceleration'],
        yaw_brake_rad_s2=command['yaw_brake'],wheel_envelope_lever=command.get('wheel_envelope_lever',0.),
        wheel_reference_speed_m_s=control['wheel_radius']*control['wheel_speed'] if 'wheel_envelope_lever' in command else 0.)

def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))

def toward(value,goal,delta):
    return value+max(-delta,min(delta,goal-value))

class CommandIntegrator:
    def __init__(self,cfg):
        self.cfg=cfg
        cfg.setdefault('backward_accel_m_s2',cfg['forward_accel_m_s2'])
        cfg.setdefault('wheel_reference_speed_m_s',0.)
        cfg.setdefault('wheel_envelope_lever',0.)
        cfg.setdefault('yaw_cruise_limit_rad_s',cfg['max_yaw_rad_s'])
        cfg.setdefault('yaw_speed_decay',3.)
        self.speed_scale=1.
        self.velocity=[0.,0.,0.]
        self.acceleration=[0.,0.,0.]
        self.yaw_target=None
    def stop(self,yaw=None):
        self.velocity[:]=[0.,0.,0.];self.acceleration[:]=[0.,0.,0.]
        if yaw is not None:self.yaw_target=wrap(yaw)
    def limits(self):
        result={k:self.cfg[k] for k in PROFILE_KEYS}
        for k in ('min_forward_m_s','max_forward_m_s','max_lateral_m_s','max_yaw_rad_s','yaw_cruise_limit_rad_s'):
            result[k]*=self.speed_scale
        result['current_yaw_rad_s']=yaw_limit(*self.velocity[:2],result['max_forward_m_s'],result['min_forward_m_s'],
            result['max_lateral_m_s'],result['max_yaw_rad_s'],result['yaw_cruise_limit_rad_s'],result['yaw_speed_decay'])
        return result
    def step(self,keys,dt,yaw=None,enabled=True):
        if dt<=0:raise ValueError('Control dt must be positive')
        if self.yaw_target is None:self.yaw_target=wrap(yaw or 0.)
        if not enabled:self.stop(yaw);return self.velocity[:]
        c=self.cfg
        signs=[int('W'in keys)-int('S'in keys),int('A'in keys)-int('D'in keys),int('Q'in keys)-int('E'in keys)]
        accel_x=c['backward_accel_m_s2'] if signs[0]<0 else c['forward_accel_m_s2']
        a=[signs[0]*accel_x,signs[1]*c['lateral_accel_m_s2'],signs[2]*c['yaw_accel_rad_s2']]
        norm=math.hypot(a[0],a[1])
        if norm>c['translation_accel_limit_m_s2']:
            scale=c['translation_accel_limit_m_s2']/norm;a[0]*=scale;a[1]*=scale
        old=self.velocity[:]
        brakes=[c['forward_brake_m_s2'],c['lateral_brake_m_s2'],c['yaw_brake_rad_s2']]
        new=[advance_axis(old[i],signs[i],dt,abs(a[i]) if signs[i] else 0.,brakes[i]) for i in range(3)]
        limits=self.limits()
        ellipse=math.hypot(new[0]/limits['max_forward_m_s'],new[1]/limits['max_lateral_m_s'])
        projected=new[:2]
        if ellipse>1:projected=[v/ellipse for v in projected]
        projected[0]=max(limits['min_forward_m_s'],projected[0])
        was_above=math.hypot(old[0]/limits['max_forward_m_s'],old[1]/limits['max_lateral_m_s'])>1.+1.e-9 or old[0]<limits['min_forward_m_s']
        # Lowering K limits requests ordinary braking on each translation axis.
        for i in range(2):
            if was_above and abs(old[i])>abs(projected[i]) and old[i]*new[i]>0:
                new[i]=brake_to_limit(new[i],old[i],abs(projected[i]),brakes[i],dt)
            else:new[i]=projected[i]
        cap=yaw_limit(*new[:2],limits['max_forward_m_s'],limits['min_forward_m_s'],
            limits['max_lateral_m_s'],limits['max_yaw_rad_s'],limits['yaw_cruise_limit_rad_s'],limits['yaw_speed_decay'])
        new[2]=brake_to_limit(new[2],old[2],cap,brakes[2],dt)
        if c['wheel_reference_speed_m_s']:
            available=max(0.,c['wheel_reference_speed_m_s']-c['wheel_envelope_lever']*abs(new[2]))
            new[0]=max(-available,min(available,new[0]))
        self.velocity[:]=new
        self.acceleration[:]=[(new[i]-old[i])/dt for i in range(3)]
        # Follow commanded rotation through release braking, then lock heading.
        self.yaw_target=wrap(self.yaw_target+.5*(old[2]+new[2])*dt)
        return new[:]

def world_to_yaw_frame(vx,vy,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return [c*vx+s*vy,-s*vx+c*vy]
