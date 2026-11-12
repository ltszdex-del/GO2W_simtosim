import json
import unittest
from pathlib import Path
from control import Controller
from low_speed_hold import LowSpeedHold

class Adapter:
    path='test://adapter'
    def __init__(self):self.resets=0
    def reset(self,state):self.resets+=1
    def is_complete(self,state):return True

class LowSpeedTests(unittest.TestCase):
    def gate(self):return LowSpeedHold(dict(enabled=True,reference_enter_speed_m_s=.1,actual_enter_speed_m_s=.4,reference_exit_speed_m_s=.1))
    def controller(self,enabled=True,ready=True):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg['low_speed_hold']=dict(enabled=enabled,reference_enter_speed_m_s=.1,actual_enter_speed_m_s=.4,reference_exit_speed_m_s=.1)
        c=Controller(cfg,lambda:0.)
        c.adapters={'locomotion':Adapter()}
        if ready:c.adapters['standing_hold']=Adapter()
        c.mode='locomotion';c.active=c.adapters['locomotion']
        return c
    def state(self,v=0.):return dict(yaw_rad=0.,actual_velocity_yaw_m_s_rad_s=[v,0.,0.])
    def test_both_speeds_and_norm(self):
        g=self.gate()
        self.assertEqual(g.select('locomotion',[.06,.06,0],[.07,0,0],True),'standing_hold')
        for ref,actual in [([.09,.09,0],[0,0,0]),([0,0,0],[.4,0,0]),([.1,0,0],[0,0,0])]:
            self.assertIsNone(g.select('locomotion',ref,actual,True))
    def test_missing_and_nonfinite_state(self):
        g=self.gate()
        for actual in (None,[float('nan'),0,0],[float('inf'),0,0]):
            self.assertIsNone(g.select('locomotion',[0,0,0],actual,True))
    def test_missing_adapter_does_not_stop_locomotion(self):
        c=self.controller(ready=False);a=c.active;c.tick(.02,self.state())
        self.assertIs(c.active,a)
        self.assertEqual(c.status()['low_speed_hold']['reason'],'standing_hold_adapter_missing')
    def test_disabled_is_inert(self):
        c=self.controller(enabled=False);a=c.active;c.tick(.02,self.state())
        self.assertIs(c.active,a);self.assertEqual(c.mode,'locomotion')
    def test_entry_and_hysteresis(self):
        c=self.controller();c.tick(.02,self.state())
        self.assertEqual(c.mode,'standing_hold')
        c.tick(.02,self.state(.11));self.assertEqual(c.mode,'standing_hold')
        c.tick(.02,self.state(2.));self.assertEqual(c.mode,'standing_hold')
        c.feed('test',['W']);c.command[0]=.08
        c.tick(.02,self.state());self.assertEqual(c.mode,'locomotion')
    def test_exit_requires_reference_only_and_strictly_above_point_one(self):
        g=self.gate()
        self.assertIsNone(g.select('standing_hold',[.1,0,0],[2.,0,0],True))
        self.assertEqual(g.select('standing_hold',[.101,0,0],None,True),'locomotion')
    def test_acceleration_wakes_without_losing_command_or_keys(self):
        c=self.controller();c.tick(.02,self.state());c.feed('test',['W'])
        for _ in range(3):c.tick(.02,self.state())
        self.assertEqual(c.mode,'locomotion');self.assertIn('W',c.keys)
        self.assertAlmostEqual(c.command[0],.12)
    def test_hold_adapter_not_treated_as_one_shot_action(self):
        c=self.controller();c.tick(.02,self.state());c.after_step(self.state())
        self.assertEqual(c.mode,'standing_hold')
    def test_u_completion_uses_same_hold_adapter(self):
        c=self.controller();c.mode='getting_up';c.active=Adapter()
        self.assertEqual(c.after_step(self.state()),'standing_hold')
        self.assertIs(c.active,c.adapters['standing_hold'])
    def test_pending_actions_and_error_do_not_switch(self):
        g=self.gate()
        for mode in ('getting_up','prone','error','lying_down','handstand'):
            self.assertIsNone(g.select(mode,[0,0,0],[0,0,0],True))
        self.assertIsNone(g.select('locomotion',[0,0,0],[0,0,0],True,pending=True))

if __name__=='__main__':unittest.main()
