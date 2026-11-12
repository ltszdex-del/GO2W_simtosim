import copy,json,math,unittest
from pathlib import Path
from commands import CommandIntegrator,world_to_yaw_frame
from control import Controller

CFG=json.loads(Path(__file__).with_name('config.json').read_text())

class CommandTests(unittest.TestCase):
    def test_six_axes_and_signed_caps(self):
        for key,index,cap in [('W',0,.6),('S',0,-.6),('A',1,.2),('D',1,-.2),('Q',2,.8),('E',2,-.8)]:
            with self.subTest(key=key):
                c=CommandIntegrator(CFG)
                for _ in range(500):c.step({key},.02,0.)
                self.assertAlmostEqual(c.velocity[index],cap)
                self.assertEqual([v for i,v in enumerate(c.velocity) if i!=index],[0.,0.])
    def test_diagonal_and_mixed(self):
        c=CommandIntegrator(CFG);c.step(set('WAQ'),.02,0.)
        self.assertLessEqual(math.hypot(*c.acceleration[:2]),.8+1e-10)
        self.assertTrue(all(v>0 for v in c.velocity))
        for _ in range(300):c.step(set('WAQ'),.02,0.)
        self.assertLessEqual(math.hypot(c.velocity[0]/.6,c.velocity[1]/.2),1.+1e-10)
        self.assertAlmostEqual(c.velocity[2],.8)
    def test_opposition_brakes_reverse_no_windup(self):
        c=CommandIntegrator(CFG)
        for _ in range(300):c.step({'W'},.02)
        c.step({'W','S'},.02);self.assertAlmostEqual(c.velocity[0],.584)
        c.step({'S'},.02);self.assertAlmostEqual(c.velocity[0],.568)
        for _ in range(100):c.step(set(),.02)
        self.assertEqual(c.velocity,[0.,0.,0.])
    def test_heading_continues_braking_then_locks(self):
        c=CommandIntegrator(CFG)
        for _ in range(100):c.step({'Q'},.02,1.)
        before=c.yaw_target;c.step({'W'},.02,1.)
        self.assertNotEqual(c.yaw_target,before)
        for _ in range(100):c.step({'W'},.02,1.)
        locked=c.yaw_target;c.step({'W'},.02,2.)
        self.assertEqual(c.yaw_target,locked)
        self.assertEqual(c.velocity[2],0.)
    def test_yaw_only_signs(self):
        bx,by=world_to_yaw_frame(0.,1.,math.pi/2)
        self.assertAlmostEqual(bx,1.);self.assertAlmostEqual(by,0.)
    def test_release_source_transfer_and_blocked_modes(self):
        now=[0.];c=Controller(CFG,lambda:now[0])
        for _ in range(50):c.feed('a',['W']);c.tick(.02)
        self.assertFalse(c.feed('b',['A']))
        c.release('a');self.assertEqual(c.command,[0.,0.,0.])
        self.assertTrue(c.feed('b',['A']));c.tick(.02)
        self.assertEqual(c.command[0],0.);self.assertGreater(c.command[1],0.)
        for mode in ['lying_down','getting_up','prone','handstand','error']:
            c.mode=mode;c.feed('b',list('WAQ'));c.tick(.02)
            self.assertEqual(c.command,[0.,0.,0.])
    def test_repeat_frequency_not_integration_clock(self):
        a,b=Controller(CFG,lambda:0.),Controller(CFG,lambda:0.)
        for _ in range(100):
            a.feed('s',['W']);a.tick(.02)
            for _ in range(10):b.feed('s',['W'])
            b.tick(.02)
        self.assertEqual(a.command,b.command)
    def test_expired_source_does_not_inherit_braking_reference(self):
        now=[0.];c=Controller(CFG,lambda:now[0])
        for _ in range(50):c.feed('a',['W']);c.tick(.02)
        c.feed('a',[]);c.tick(.02)
        self.assertGreater(c.command[0],0.)
        now[0]=.5
        self.assertTrue(c.feed('b',['A']));c.tick(.02)
        self.assertEqual(c.command[0],0.)
        self.assertGreater(c.command[1],0.)

class Adapter:
    path='test://never-production'
    def __init__(self,manifest):self.manifest=manifest;self.calls=[]
    def load(self):self.calls.append('load')
    def reset(self,state):self.calls.append('reset')
    def observe(self,state):self.calls.append('observe');return state
    def infer(self,obs):self.calls.append('infer');return [0.]*16
    def apply(self,action,dt):self.calls.append('apply');return {'leg_pos_rad':[0.,.8,-1.5]*4,'wheel_vel_rad_s':[0.]*4}
    def is_complete(self,state):self.calls.append('is_complete');return True

class AdapterTests(unittest.TestCase):
    def test_contract_execution_and_continuous_handoff(self):
        c=Controller(CFG,lambda:0.)
        c.contract={k:{} for k in ['joint_order','output','pd','limits','timing','asset_sha256']}
        manifest={**c.contract,**{k:None for k in ['observation_order','normalization','scales','history','phase','previous_action','target','task_id']}}
        a=Adapter(manifest);c.register('lying_down',a)
        state={'yaw_rad':0.,'actual_velocity_yaw_m_s_rad_s':[0.,0.,0.]}
        c.feed('s',['P']);c.tick(.02,state)
        for _ in range(20):c.tick(.02,state)
        self.assertEqual(c.mode,'lying_down')
        command=c.execute(state,.02);self.assertEqual(len(command['leg_pos_rad']),12)
        self.assertEqual(c.after_step(state),'prone')
        self.assertEqual(a.calls,['load','reset','observe','infer','apply','is_complete'])
        bad=Adapter(copy.deepcopy(manifest));bad.manifest['asset_sha256']='other'
        with self.assertRaisesRegex(ValueError,'asset_sha256'):c.register('getting_up',bad)
    def test_p_waits_for_braking_and_actual_settling(self):
        c=Controller(CFG,lambda:0.)
        c.adapters['lying_down']=Adapter({})
        c.feed('s',['W']);c.tick(.2,{'yaw_rad':0.,'actual_velocity_yaw_m_s_rad_s':[.2,0.,0.]})
        c.feed('s',['P']);c.tick(.02)
        for _ in range(50):c.tick(.02)
        self.assertEqual(c.mode,'standing');self.assertEqual(c.pending,'lying_down')
        for _ in range(20):c.tick(.02,{'yaw_rad':0.,'actual_velocity_yaw_m_s_rad_s':[0.,0.,0.]})
        self.assertEqual(c.mode,'lying_down');self.assertEqual(c.command,[0.,0.,0.])
if __name__=='__main__':unittest.main()
