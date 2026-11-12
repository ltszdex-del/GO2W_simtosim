import unittest,torch
from schedule import Commands,Schedule
from specialized_schedule import SpecializedSchedule,specialized_phase
from config import COMMAND
class Tests(unittest.TestCase):
    def test_distribution(self):
        s=SpecializedSchedule(1024,'cpu',42)
        self.assertEqual(torch.bincount(s.group).tolist(),[205,103,103,103,102,102,102,102,102])
    def test_all_scenarios_and_speed_bounds(self):
        s=SpecializedSchedule(1024,'cpu',42);s.reset(torch.arange(1024))
        c=Commands(1024,'cpu');positive=torch.zeros(9,2,dtype=torch.bool);negative=positive.clone();rest=positive.clone()
        last=None;maximum=torch.zeros(2);minimum=torch.zeros(2)
        for i in range(600):
            age=torch.full((1024,),i*.02);a=s.tick(2,age,.02);v=c.tick(a,.02)
            self.assertTrue(torch.isfinite(v).all())
            self.assertLessEqual(float((v[:,:2]/torch.tensor([2.3,1.3])).norm(dim=-1).max()),1.00001)
            maximum=torch.maximum(maximum,v[:,:2].amax(0));minimum=torch.minimum(minimum,v[:,:2].amin(0))
            for k in range(9):
                x=a[s.group==k,:2];positive[k]|=(x>0).any(0);negative[k]|=(x<0).any(0);rest[k]|=(x==0).any(0)
            if i==448:last=v.clone()
        self.assertFalse(positive[0].any());self.assertTrue(negative[0,0])
        self.assertTrue(positive[7,0] and negative[7,0]);self.assertTrue(positive[8,1] and negative[8,1])
        self.assertTrue(rest[5,0] and rest[6,1]);self.assertTrue(torch.allclose(v,torch.zeros_like(v)))
        self.assertTrue(torch.allclose(maximum,torch.tensor([2.3,1.3]),atol=1e-5))
        self.assertTrue(torch.allclose(minimum,torch.tensor([-2.3,-1.3]),atol=1e-5))
    def test_maximum_brakes_within_half_second(self):
        c=Commands(4,'cpu');c.velocity[:,:2]=torch.tensor([[2.3,0],[-2.3,0],[0,1.3],[0,-1.3]])
        for _ in range(25):c.tick(torch.zeros(4,3),.02)
        self.assertTrue(torch.allclose(c.velocity,torch.zeros_like(c.velocity),atol=1e-5))
    def test_exact_phase_boundary(self):
        self.assertTrue(specialized_phase(4200));self.assertTrue(specialized_phase(4399))
        self.assertFalse(specialized_phase(4400));self.assertFalse(specialized_phase(4799))
if __name__=='__main__':unittest.main()
