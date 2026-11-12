import json
import unittest
from pathlib import Path
from control import Controller

class Adapter:
    path='test://handstand'
    def __init__(self):self.resets=0;self.ready=False
    def reset(self,state):self.resets+=1
    def ready_to_start(self,state):return self.ready
    def is_complete(self,state):return False

class HandstandSwitchTests(unittest.TestCase):
    def setUp(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        self.c=Controller(cfg,lambda:0.)
        self.h=Adapter();self.move=Adapter()
        self.c.adapters={'locomotion':self.move,'handstand':self.h}
        self.c.mode='locomotion';self.c.active=self.move
    def state(self,speed=0.):return dict(yaw_rad=0.,actual_velocity_yaw_m_s_rad_s=[speed,0.,0.])
    def press_h(self):self.c.feed('test',['H']);self.c.feed('test',[])
    def test_moving_robot_brakes_before_stance_handoff(self):
        self.press_h();self.c.command[:]=[1.,0.,0.]
        self.c.tick(.02,self.state(.8))
        self.assertIs(self.c.active,self.move);self.assertEqual(self.c.pending,'handstand')
        self.assertAlmostEqual(self.c.command[0],.92)
    def test_wait_for_reference_pose_then_activate_once(self):
        self.press_h()
        for _ in range(25):self.c.tick(.02,self.state())
        self.assertEqual(self.h.resets,0);self.assertIsNone(self.c.active)
        self.h.ready=True
        for _ in range(16):self.c.tick(.02,self.state())
        self.assertEqual(self.h.resets,1);self.assertEqual(self.c.mode,'handstand')
        self.press_h();self.c.tick(.02,self.state())
        self.assertEqual(self.h.resets,1)
    def test_failure_disables_policy_without_auto_reset(self):
        self.c.active=self.h;self.c.mode='handstand'
        self.h.after_physics=lambda state:True;self.h.failure='collision'
        self.assertEqual(self.c.after_step(self.state()),'error')
        self.assertIsNone(self.c.active);self.assertEqual(self.c.mode,'error')

if __name__=='__main__':unittest.main()
