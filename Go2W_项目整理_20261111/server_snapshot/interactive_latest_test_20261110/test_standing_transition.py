import unittest,json
from pathlib import Path
from standing_transition import StandingTransition
from control import Controller
class Adapter:
    path='test'
    def reset(self,state):pass
def state(speed=0.,tilt=0.,contacts=True):
    return dict(yaw_rad=0.,actual_velocity_yaw_m_s_rad_s=[speed,0,0],
        standing_readiness=dict(roll_deg=tilt,pitch_deg=0.,angular_speed_rad_s=.05,four_contacts=contacts,height_m=.39))
class Tests(unittest.TestCase):
    def test_dwell_and_break_resets_timer(self):
        t=StandingTransition()
        for _ in range(9):self.assertFalse(t.ready(state()))
        self.assertFalse(t.ready(state(tilt=5.1)));self.assertEqual(t.stable_s,0.)
        for _ in range(9):self.assertFalse(t.ready(state()))
        self.assertTrue(t.ready(state()))
    def test_unsafe_states_never_time_out_to_hold(self):
        for s in [state(speed=.3),state(tilt=8.),state(contacts=False),{}]:
            t=StandingTransition()
            for _ in range(100):self.assertFalse(t.ready(s))
    def controller(self):
        c=Controller(json.loads(Path(__file__).with_name('config.json').read_text()),lambda:0.)
        c.adapters={k:Adapter() for k in ('locomotion','standing_hold')}
        c.mode='locomotion';c.active=c.adapters['locomotion'];return c
    def test_wait_under_backward_braking_and_recover(self):
        c=self.controller()
        for _ in range(30):c.tick(.02,state(speed=-.2,tilt=8.))
        self.assertEqual(c.mode,'locomotion');self.assertEqual(c.transition.phase,'braking')
        for _ in range(10):c.tick(.02,state())
        self.assertEqual(c.mode,'standing_hold')
        c.feed('test',['S'])
        for _ in range(3):c.tick(.02,state())
        self.assertEqual(c.mode,'locomotion');self.assertLess(c.command[0],-.1)
    def test_new_input_cancels_wait(self):
        c=self.controller();c.tick(.02,state(tilt=8.));c.feed('test',['W'])
        for _ in range(3):c.tick(.02,state(tilt=8.))
        self.assertEqual(c.mode,'locomotion');self.assertEqual(c.transition.phase,'idle')
if __name__=='__main__':unittest.main()
