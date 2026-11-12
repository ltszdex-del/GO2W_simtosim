"""Regression: a native key held without repeat events must reach and retain the cap."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from control import Controller
from keyboard_router import KeyboardRouter

class HeldKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.c=Controller(json.loads(Path(__file__).with_name('config.json').read_text()),lambda:self.now)
        self.k=KeyboardRouter(self.c,lambda:None)
    def event(self,key,kind='KEY_PRESS'):self.k.handle(NS(input=key,type=kind))
    def tick(self,down):
        self.now+=.05  # Slow rendering: timeout is wall time, integration is simulation time.
        self.k.poll(lambda key:key in down)
        self.c.tick(.02)
    def test_hold_once_reaches_cap_and_keeps_it(self):
        self.event('W')
        for _ in range(200):self.tick({'W'})
        self.assertAlmostEqual(self.c.command[0],2.)
        self.assertEqual(self.c.stats['timeouts'],0)
        for _ in range(80):
            self.tick({'W'})
            self.assertAlmostEqual(self.c.command[0],2.)
        self.event('W','KEY_RELEASE')
        self.tick(set());self.assertAlmostEqual(self.c.command[0],1.92)
        for _ in range(24):self.tick(set())
        self.assertEqual(self.c.command,[0.,0.,0.])
    def test_polled_release_without_event_brakes(self):
        self.event('W')
        for _ in range(100):self.tick({'W'})
        self.tick(set())
        self.assertAlmostEqual(self.c.command[0],1.92)
        self.assertFalse(self.c.keys)
    def test_render_stall_does_not_zero_verified_held_input(self):
        self.event('W')
        for _ in range(100):self.tick({'W'})
        self.now+=.8
        self.tick({'W'})
        self.assertAlmostEqual(self.c.command[0],2.)
        self.assertEqual(self.c.stats['timeouts'],0)
    def test_render_stall_with_release_still_brakes(self):
        self.event('W')
        for _ in range(100):self.tick({'W'})
        self.now+=.8;self.tick(set())
        self.assertAlmostEqual(self.c.command[0],1.92)
    def test_repeat_event_after_render_stall_keeps_reference(self):
        self.event('W')
        for _ in range(100):self.tick({'W'})
        self.now+=.8;self.event('W','KEY_REPEAT');self.tick({'W'})
        self.assertAlmostEqual(self.c.command[0],2.)
    def test_query_failure_preserves_timeout(self):
        self.event('W');self.tick({'W'})
        self.now+=.5
        def failed(_):raise RuntimeError('Device unavailable')
        self.k.poll(failed);self.c.tick(.02)
        self.assertEqual(self.c.command,[0.,0.,0.])
        self.assertEqual(self.c.stats['timeouts'],1)
    def test_stop_does_not_restart_from_stale_held_key(self):
        self.event('W');self.tick({'W'})
        self.c.release('webrtc-keyboard')
        for _ in range(30):self.tick({'W'})
        self.assertEqual(self.c.command,[0.,0.,0.])
        self.assertFalse(self.k.held)
    def test_bridge_owner_is_not_overridden(self):
        self.c.feed('bridge:test',['S'])
        self.k.held={'W'};self.tick({'W'})
        self.assertEqual(self.c.source,'bridge:test')
        self.assertLess(self.c.command[0],0.)
    def test_release_one_axis_keeps_other(self):
        self.event('W');self.event('A')
        for _ in range(50):self.tick({'W','A'})
        for _ in range(30):self.tick({'A'})
        self.assertEqual(self.c.command[0],0.)
        self.assertGreater(self.c.command[1],0.)
        self.assertEqual(self.c.stats['timeouts'],0)

if __name__=='__main__':unittest.main()
