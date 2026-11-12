import json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from control import Controller
from keyboard_router import KeyboardRouter

class KeyboardTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.opens=0
        self.c=Controller(json.loads(Path(__file__).with_name('config.json').read_text()),lambda:self.now)
        self.k=KeyboardRouter(self.c,self.open)
    def open(self):self.opens+=1
    def event(self,key,kind='KEY_PRESS'):
        return self.k.handle(NS(input=key,type=kind))
    def test_f1_and_enum(self):
        self.event(NS(name='F1'));self.assertEqual(self.opens,1)
        self.event(NS(name='UP'));self.c.tick(.1);self.assertGreater(self.c.command[0],0)
    def test_text_repeat_holds_cap_and_release_zero(self):
        self.event('W')
        for _ in range(200):
            self.now+=.02;self.event('w','CHAR');self.c.tick(.02)
        self.assertAlmostEqual(self.c.command[0],self.c.cfg['max_forward_m_s'])
        self.assertEqual(self.c.stats['timeouts'],0)
        self.event('W','KEY_RELEASE');self.c.tick(.02);self.assertAlmostEqual(self.c.command[0],self.c.cfg['max_forward_m_s']-.02*self.c.cfg['forward_brake_m_s2'])
        self.c.release('webrtc-keyboard');self.assertEqual(self.c.command,[0.,0.,0.])
    def test_loss_of_events_still_times_out(self):
        self.event('W');self.c.tick(.02);self.now+=.41;self.c.tick(.02)
        self.assertEqual(self.c.command,[0.,0.,0.]);self.assertIn('TIMED OUT',self.c.input_state)
    def test_action_text_repeat_no_retrigger(self):
        self.event('H');self.c.tick(.02)
        for _ in range(10):self.event('h','CHAR');self.c.tick(.02)
        self.assertEqual(self.c.stats['trigger_H'],1)
    def test_yaw_cap(self):
        for _ in range(200):
            self.now+=.02;self.event('Q','KEY_REPEAT');self.c.tick(.02)
        self.assertAlmostEqual(self.c.command[2],self.c.cfg['max_yaw_rad_s'])
if __name__=='__main__':unittest.main()
