"""Measured-state dwell gate; no simulator dependency and no timeout force-switch."""
import math
class StandingTransition:
    def __init__(self):self.cancel()
    def cancel(self):self.stable_s=0.;self.settle_s=0.;self.phase='idle';self.reason=''
    def ready(self,state,dt=.02):
        v=state.get('actual_velocity_yaw_m_s_rad_s',[math.inf]*3)
        m=state.get('standing_readiness',{})
        checks={'speed':math.hypot(*v[:2])<.1,
            'reference_yaw':abs(v[2])<.2,
            'tilt':max(abs(m.get('roll_deg',math.inf)),abs(m.get('pitch_deg',math.inf)))<3.,
            'angular_speed':m.get('angular_speed_rad_s',math.inf)<.2,
            'four_contacts':m.get('four_contacts',False),
            'height':.36<m.get('height_m',0)<.43}
        self.phase='braking';self.reason=','.join(k for k,ok in checks.items() if not ok)
        self.stable_s=self.stable_s+dt if all(checks.values()) else 0.
        if self.stable_s>=.2-1e-8:self.phase='ready';return True
        return False
    def status(self):return dict(phase=self.phase,stable_s=round(self.stable_s,3),settle_s=round(self.settle_s,3),waiting_for=self.reason)
