import unittest,torch
from schedule import Schedule,Commands
from config import GAIT,COMMAND

class Tests(unittest.TestCase):
    def test_stage2_balanced_axes_reversals_and_stops(self):
        s=Schedule(1024,'cpu',45678);ids=torch.arange(1024);s.reset(ids)
        self.assertEqual(torch.bincount(s.primary).tolist(),[512,512])
        counts=torch.zeros(4);zero=0;flips=0;last=torch.zeros(1024)
        c=Commands(1024,'cpu')
        for step in range(600):
            age=torch.full((1024,),step*.02);a=s.tick(2,age,.02);v=c.tick(a,.02)
            self.assertTrue(bool(((a[:,:2]!=0).sum(-1)<=1).all()))
            self.assertTrue(bool((a[:,2]==0).all()))
            self.assertTrue(bool((v[:,0]>=-1.500001).all()))
            for g,(axis,sign) in enumerate(((0,1),(0,-1),(1,1),(1,-1))):counts[g]+=(a[:,axis]==sign).sum()
            signed=a[:,:2].sum(-1);flips+=int((signed*last<0).sum());zero+=int((signed==0).sum());last=signed
        self.assertGreater(counts.min().item(),20000)
        self.assertLess(counts.max().item()/counts.min().item(),1.2)
        self.assertGreater(flips,200);self.assertGreater(zero,10000)
        self.assertLess(v.abs().max().item(),1e-5)
    def test_preserved_gait_and_limits(self):
        self.assertEqual(COMMAND['min_forward_speed'],-1.5)
        self.assertEqual(COMMAND['speed_limits'],[2.3,1.3])
        self.assertEqual(GAIT['period'],.3)
        self.assertEqual(GAIT['failure_penalty'],2.5)
        self.assertEqual(GAIT['front_fold_scope'],'all_translation')
