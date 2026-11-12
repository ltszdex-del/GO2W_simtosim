import json,unittest
from pathlib import Path
from control import Controller
from stream_ui import StreamPanel

class VirtualKeyboardTests(unittest.TestCase):
    def setUp(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        self.c=Controller(cfg)
        self.p=StreamPanel.__new__(StreamPanel)
        self.p.controller=self.c;self.p.deadline=0.;self.p.notice=''
    def test_virtual_motion_and_release(self):
        self.p.press('W');self.c.tick(.1)
        self.assertGreater(self.c.command[0],0)
        self.assertIn('RECEIVED W',self.p.notice)
        self.p.release();self.assertEqual(self.c.command,[0.,0.,0.])
    def test_virtual_actions(self):
        for key in 'HPU':
            self.p.press(key);self.c.tick(.02);self.p.release()
            self.assertEqual(self.c.stats['trigger_'+key],1)
            self.assertEqual(self.c.last_trigger,key)
            self.assertEqual(self.c.mode,'standing')
    def test_busy_source(self):
        self.c.feed('browser',['W']);self.p.press('S')
        self.assertIn('BUSY',self.p.notice)
        self.assertEqual(self.c.keys,{'W'})
if __name__=='__main__':unittest.main()
