"""Read-only paired PhysX slip evaluation; writes only a separate evaluation folder."""
import argparse,json,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT=Path('/home/ubuntu/Go2W/go2w_handstand_targeted4179_20261112');sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args()
launcher=AppLauncher(a);app=launcher.app;env=None
folder=Path('/home/ubuntu/Go2W/evaluations/handstand_targeted20s_20261112');folder.mkdir(parents=True,exist_ok=True)
try:
    import torch
    from go2w_mimic.env import MimicEnv
    from go2w_mimic.env_cfg import MimicEnvCfg
    from go2w_mimic.networks import Teacher,Student
    from go2w_mimic.settings import MOTION
    torch.set_num_threads(4);torch.manual_seed(42)
    teacher_path=Path('/home/ubuntu/Go2W/go2w_handstand_currentU80_20261108/logs/train80/teacher_4179.pt')
    teacher=Teacher().to('cuda:0').eval();teacher.load_state_dict(torch.load(teacher_path,map_location='cpu',weights_only=False)['model'])
    student=Student().to('cuda:0').eval();student.load_state_dict(torch.load(ROOT/'logs/targeted1000/student_999.pt',map_location='cpu',weights_only=False)['model'])
    cfg=MimicEnvCfg();cfg.scene.num_envs=256;cfg.sim.device='cuda:0';cfg.seed=42;cfg.motion_path=str(MOTION)
    cfg.randomize=True;cfg.domain_level=.2;cfg.episode_length_s=40.
    env=MimicEnv(cfg);env.cfg.randomize=False;results={}
    with torch.inference_mode():
        for name,model in [('teacher',teacher),('student',student)]:
            torch.manual_seed(18217);obs,_=env.reset();alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            windows={w:dict(n=0,slip_sq=0.,axle_speed_sq=0.,anchor_error_sq=0.,wheel_dq_sq=0.,gravity_x=0.,action_mse=0.) for w in ['common_hold','late']}
            good=torch.zeros(env.num_envs,device=env.device)
            for k in range(1000):
                if 125<=k<200 or k>=250:
                    w=windows['common_hold' if k<200 else 'late'];phys=env.physical();d=env.robot.data;mask=alive
                    if bool(mask.any()):
                        w['n']+=1
                        values=dict(slip_sq=phys['slip'][mask,:2].mean(),
                            axle_speed_sq=phys['axle_velocity_xy'][mask,:2].square().sum(-1).mean(),
                            anchor_error_sq=env.anchor_error(phys)[mask].square().sum(-1).mean(),
                            wheel_dq_sq=d.joint_vel.torch[mask][:,env.wheel_ids[:2]].square().mean(),
                            gravity_x=d.projected_gravity_b.torch[mask,0].mean(),
                            action_mse=(student(obs['student'])[mask]-teacher(obs['policy'])[mask]).square().mean())
                        for key,value in values.items():w[key]+=float(value)
                action=model(obs['policy' if name=='teacher' else 'student'])
                obs,_,term,trunc,_=env.step(action);alive&=~(term|trunc)
                if k>=250:good+=(env.robot.data.projected_gravity_b.torch[:,0]>.9).float()
            results[name]=dict(survival_rate=float(alive.float().mean()),
                success_rate=float((alive&(good/750>.95)).float().mean()),windows=windows)
            for w in windows.values():
                n=max(w.pop('n'),1)
                for key in list(w):w[key]/=n
                for key in ['slip_sq','axle_speed_sq','anchor_error_sq','wheel_dq_sq']:w[key.replace('_sq','_rms')]=w.pop(key)**.5
            (folder/'results.json').write_text(json.dumps(results,indent=2))
            print('[SLIP_EVAL] '+json.dumps({name:results[name]}),flush=True)
except BaseException:
    traceback.print_exc();raise
finally:
    if env is not None:env.close()
    app.close()
