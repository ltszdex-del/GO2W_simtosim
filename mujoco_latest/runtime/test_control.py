import json
from pathlib import Path
import unittest
from control import Controller

class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.now=10.
        self.c=Controller(json.loads(Path(__file__).with_name('config.json').read_text()),lambda:self.now)
    def test_release_and_timeout(self):
        self.c.feed('browser',['W','A']);self.c.tick(.1)
        self.assertGreater(self.c.command[0],0)
        self.c.feed('browser',[]);self.c.tick(.02)
        self.assertGreater(self.c.command[0],0)
        self.c.release('browser')
        self.assertEqual(self.c.command,[0.,0.,0.])
        self.c.feed('browser',['S']);self.c.tick(.1)
        self.now+=.41;self.c.tick(.02)
        self.assertEqual(self.c.command,[0.,0.,0.])
        self.assertEqual(self.c.stats['timeouts'],1)
    def test_trigger_edges_and_unloaded_policy(self):
        for _ in range(20): self.c.feed('browser',['P']);self.c.tick(.02)
        self.assertEqual(self.c.stats['trigger_P'],1)
        self.assertEqual(self.c.mode,'standing')
        self.assertIn('未加载',self.c.message)
        self.c.feed('browser',[]);self.c.tick(.02)
        self.c.feed('browser',['P']);self.c.tick(.02)
        self.assertEqual(self.c.stats['trigger_P'],2)
    def test_focus_loss_and_exclusive_source(self):
        self.c.feed('a',['W']);self.c.tick(.1)
        self.assertFalse(self.c.feed('b',['S']))
        self.c.release('a')
        self.assertEqual(self.c.command,[0.,0.,0.])
        self.assertTrue(self.c.feed('b',['S']))
    def test_invalid_transition_and_ramp(self):
        self.c.feed('a',['U']);self.c.tick(.02)
        self.assertIn('需要 prone',self.c.message)
        self.c.feed('a',['W']);self.c.tick(.02)
        self.assertAlmostEqual(self.c.command[0],.016)
        self.c.mode='handstand'
        self.c.feed('a',['P']);self.c.tick(.02)
        self.assertEqual(self.c.mode,'handstand')
        self.assertIn('不支持',self.c.message)

if __name__=='__main__': unittest.main()
