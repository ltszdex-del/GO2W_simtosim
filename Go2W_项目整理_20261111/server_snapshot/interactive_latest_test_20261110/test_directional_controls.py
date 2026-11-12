import importlib.util
import json
import math
import random
import unittest
from pathlib import Path
import torch
from commands import CommandIntegrator
from directional_reference import advance_axis,planar_tensor

class DirectionalTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    def test_lateral_cap_and_diagonal_ellipse(self):
        c=CommandIntegrator(self.cfg)
        for _ in range(200):c.step({'A'},.02)
        self.assertAlmostEqual(c.velocity[1],1.)
        for _ in range(100):c.step({'W','A'},.02)
        self.assertLessEqual(math.hypot(c.velocity[0]/2,c.velocity[1]),1.+1e-8)
    def test_max_release_stops_in_half_second_both_axes(self):
        c=CommandIntegrator(self.cfg)
        for initial in ([2.,0.,0.],[0.,-1.,0.]):
            c.velocity[:]=initial
            for _ in range(25):c.step(set(),.02)
            self.assertEqual(c.velocity,[0.,0.,0.])
    def test_reverse_is_faster_than_release(self):
        for velocity,accel,brake in [(2.,2.,4.),(-2.,2.,4.),(1.,1.,2.),(-1.,1.,2.)]:
            axis=-math.copysign(1.,velocity)
            reverse=advance_axis(velocity,axis,.02,accel,brake)
            coast=advance_axis(velocity,0.,.02,accel,brake)
            self.assertLess(abs(reverse),abs(coast))
            self.assertAlmostEqual(advance_axis(velocity,axis,1/3,accel,brake),0.,places=7)
    def test_crossing_zero_only_uses_input_after_crossing(self):
        self.assertAlmostEqual(advance_axis(.03,-1.,.02,2.,4.),-.03)
        self.assertAlmostEqual(advance_axis(-.03,1.,.02,2.,4.),.03)
    def test_5000_steps_training_and_keyboard_match(self):
        c=CommandIntegrator(self.cfg);v=torch.zeros(1,2);rng=random.Random(14)
        for i in range(5000):
            if i%17==0:keys=set(rng.sample(list('WASD'),rng.randrange(5)))
            axes=torch.tensor([[int('W' in keys)-int('S' in keys),int('A' in keys)-int('D' in keys)]],dtype=torch.float)
            v=planar_tensor(v,axes,.02,[2.,1.],[4.,2.],[2.,1.])
            expected=c.step(keys,.02)
            self.assertTrue(torch.allclose(v,torch.tensor([expected[:2]]),atol=4e-6),(i,v,expected))

if __name__=='__main__':unittest.main()
