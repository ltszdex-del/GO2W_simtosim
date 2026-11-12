import unittest,torch,math
from schedule import Commands,wrap
from directional_reference import advance_axis
from qe_schedule import MixedSchedule

class Tests(unittest.TestCase):
    def test_yaw_integrator(self):
        c=Commands(1,'cpu');v=heading=0.;g=torch.Generator().manual_seed(9)
        for k in range(1000):
            if k%41==0:axis=int(torch.randint(-1,2,(1,),generator=g))
            old=v;v=max(-.8,min(.8,advance_axis(v,axis,.02,1.,1.6)))
            heading=math.atan2(math.sin(heading+.01*(old+v)),math.cos(heading+.01*(old+v)))
            result=c.tick(torch.tensor([[0.,0.,axis]]),.02)
            self.assertAlmostEqual(float(result[0,2]),v,places=5)
            self.assertAlmostEqual(float(wrap(c.heading[0]-heading)),0.,places=4)
        c.velocity[0,2]=.8
        for _ in range(25):c.tick(torch.zeros(1,3),.02)
        self.assertLess(float(c.velocity[0,2].abs()),1e-5)
    def test_partition_and_combination(self):
        for stage in (1,2):
            s=MixedSchedule(1024,'cpu',89,stage);s.reset(torch.arange(1024));combined=0
            self.assertEqual(int(s.qe_mask.sum()),512)
            for k in range(500):
                a=s.tick(3,torch.full((1024,),k*.02),.02)
                self.assertTrue(bool((a[~s.qe_mask,2]==0).all()))
                if stage==1:self.assertTrue(bool((a[s.qe_mask,:2]==0).all()))
                combined+=int(((a[:,:2].abs().sum(-1)>0)&(a[:,2]!=0)).sum())
            self.assertEqual(combined,0) if stage==1 else self.assertGreater(combined,10000)
