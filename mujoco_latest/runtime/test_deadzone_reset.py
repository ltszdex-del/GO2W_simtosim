import unittest,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from control import Controller
from low_speed_hold import LowSpeedHold
from pu_adapter import PUAdapter

class Tests(unittest.TestCase):
    def test_yaw_wakes_and_prevents_hold(self):
        g=LowSpeedHold({'enabled':True})
        self.assertEqual(g.select('standing_hold',[0,0,.2],[0,0,0],True),'locomotion')
        self.assertIsNone(g.select('locomotion',[0,0,.2],[0,0,0],True))
    def test_reset_overrides_error_and_input_lease(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        c=Controller(cfg,lambda:0.)
        c.feed('old',['W']);c.mode='error';c.pending='handstand'
        self.assertTrue(c.feed('new',['R']))
        self.assertTrue(c.reset_requested)
        c.reset_runtime({'yaw_rad':.5})
        self.assertEqual(c.mode,'standing');self.assertIsNone(c.pending)
        self.assertFalse(c.reset_requested);self.assertEqual(c.command,[0,0,0])
        self.assertTrue(c.feed('new',['W']));c.tick(.02)
        self.assertGreater(c.command[0],0)
    def test_standing_displacement_guard_only(self):
        for mode,tilt,expected in [('standing_hold',False,False),('standing_hold',True,False),('lying_down',False,True)]:
            a=PUAdapter.__new__(PUAdapter);a.mode=mode;a.task='P' if mode=='lying_down' else 'U';a.handoff_age=0
            a.ctx=NS(episode_length_buf=torch.zeros(1),failed=torch.ones(1,dtype=torch.bool),
                reason_masks={'drift':torch.tensor([True]),'body_tilt':torch.tensor([tilt])},_get_dones=lambda:None)
            self.assertEqual(a.after_physics({}),expected)

if __name__=='__main__':unittest.main()
