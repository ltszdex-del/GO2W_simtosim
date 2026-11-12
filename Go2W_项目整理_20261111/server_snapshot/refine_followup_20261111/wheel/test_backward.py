import unittest,torch
from schedule import Commands
from backward_schedule import BackwardSchedule

class Tests(unittest.TestCase):
    def test_limits_and_release(self):
        c=Commands(4,'cpu');u=torch.tensor([[-1.,0,0],[1.,0,0],[0,1.,0],[0,-1.,0]])
        for _ in range(300):v=c.tick(u,.02)
        torch.testing.assert_close(v[:,:2],torch.tensor([[-1.5,0],[2.3,0],[0,1.3],[0,-1.3]]))
        for _ in range(25):v=c.tick(torch.zeros_like(u),.02)
        self.assertLess(v.abs().max().item(),1e-5)
    def test_mixed_commands(self):
        c=Commands(1024,'cpu');g=torch.Generator().manual_seed(12)
        for i in range(500):
            if i%25==0:u=torch.randint(-1,2,(1024,3),generator=g).float();u[:,2]=0
            v=c.tick(u,.02)
            self.assertTrue(bool((v[:,0]>=-1.500001).all()))
            self.assertTrue(bool((v[:,0]<=2.300001).all()))
            self.assertTrue(bool((v[:,1].abs()<=1.300001).all()))
            self.assertTrue(bool(((v[:,:2]/torch.tensor([2.3,1.3])).norm(dim=1)<=1.000001).all()))
    def test_backward_schedule(self):
        s=BackwardSchedule(1024,'cpu',123);ids=torch.arange(1024);s.reset(ids)
        self.assertEqual(torch.bincount(s.group).tolist(),[512,256,256])
        c=Commands(1024,'cpu');active=torch.zeros(3)
        for k in range(600):
            a=s.tick(2,torch.full((1024,),k*.02),.02);v=c.tick(a,.02)
            self.assertTrue(bool((a[:,0]<=0).all() and (a[:,1:]==0).all()))
            self.assertTrue(bool((v[:,0]<=0).all()))
            for group in range(3):active[group]+=(a[s.group==group,0]<0).float().mean()
        self.assertTrue(bool((active>50).all()))
        self.assertLess(v.abs().max().item(),1e-5)
