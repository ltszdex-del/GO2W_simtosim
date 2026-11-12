"""Cross-engine DAgger refinement; original teacher, observation and control stay fixed."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=1000);p.add_argument('--num_envs',type=int,default=2048)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
launcher=AppLauncher(args);app=launcher.app;env=None
folder=ROOT/'logs/targeted1000';folder.mkdir(parents=True,exist_ok=True)
try:
    import numpy as np
    import torch
    from go2w_mimic.env import MimicEnv
    from go2w_mimic.env_cfg import MimicEnvCfg
    from go2w_mimic.networks import Teacher,Student
    from go2w_mimic.interface import manifest,verify
    from go2w_mimic.settings import MOTION
    from go2w_mimic.learning import save
    torch.set_num_threads(4);torch.manual_seed(47179);dev=args.device or 'cuda:0'
    teacher_path=Path('/home/ubuntu/Go2W/go2w_handstand_currentU80_20261108/logs/train80/teacher_4179.pt')
    assert hashlib.sha256(teacher_path.read_bytes()).hexdigest()=='6a8978ed66ebd850072a689d6f57694946e818fd812fb835c8a478be30d65eaa'
    ck=torch.load(teacher_path,map_location='cpu',weights_only=False);interface=manifest(MOTION);verify(ck['interface'],interface)
    teacher=Teacher().to(dev).eval();teacher.load_state_dict(ck['model']);teacher.requires_grad_(False)
    student=Student().to(dev)
    warm=torch.load(ROOT/'resume/student_999.pt',map_location='cpu',weights_only=False);verify(warm['interface'],interface)
    student.load_state_dict(warm['model']);optimizer=torch.optim.Adam(student.parameters(),lr=1e-4)
    data=np.load(ROOT/'mujoco_targeted4179.npz');receipt=json.loads((ROOT/'targeted_dataset.json').read_text())
    assert hashlib.sha256((ROOT/'mujoco_targeted4179.npz').read_bytes()).hexdigest()==receipt['sha256']
    fixed_x=torch.as_tensor(data['observations'],device=dev);fixed_y=torch.as_tensor(data['targets'],device=dev)
    fixed_b=torch.as_tensor(data['bucket'],device=dev);held=torch.as_tensor(data['episode']==2,device=dev)
    assert bool(held.any()) and bool((~held).any())
    offline=[torch.where((fixed_b==i)&~held)[0] for i in range(4)]
    all_train=torch.where(~held)[0];offline=[v if len(v) else all_train for v in offline]
    cfg=MimicEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=dev;cfg.seed=47179
    cfg.motion_path=str(MOTION);cfg.randomize=True;cfg.domain_level=.2;cfg.episode_length_s=12.
    env=MimicEnv(cfg);obs,_=env.reset()
    capacity=65536
    replay_x=[torch.empty(capacity,2657,dtype=torch.float16,device=dev) for _ in range(4)]
    replay_y=[torch.empty(capacity,16,device=dev) for _ in range(4)]
    cursors=[0]*4;counts=[0]*4
    weights=torch.tensor([2.]*6+[1.]*6+[6.,6.,2.,2.],device=dev);weights/=weights.mean()
    progress=(folder/'progress.jsonl').open('w')
    first=0
    for iteration in range(args.iterations):
        start=time.perf_counter();beta=.5*max(0.,1.-iteration/300.)
        with torch.inference_mode():
            for _ in range(32):
                labels=teacher(obs['policy']);pred=student(obs['student'])
                age=env.episode_length_buf.float()*.02;physical=env.physical()
                drift=env.anchor_error(physical).norm(dim=-1).amax(-1)
                disagreement=(labels[:,12:14]-pred[:,12:14]).square().mean(-1).sqrt()
                hard=(age>2.)&((drift>.12)|(disagreement>.04))
                bucket=torch.where(age<1.5,0,torch.where(age<3.,1,2))
                # Retain hard states in their original phase too; do not empty hold replay.
                hard_partition=(torch.arange(args.num_envs,device=dev)+env.episode_length_buf)%2==0
                bucket[hard&hard_partition]=3
                for i in range(4):
                    ids=torch.where(bucket==i)[0];n=len(ids)
                    if n:
                        slots=(torch.arange(n,device=dev)+cursors[i])%capacity
                        replay_x[i][slots]=obs['student'][ids].half();replay_y[i][slots]=labels[ids]
                        cursors[i]=(cursors[i]+n)%capacity;counts[i]=min(capacity,counts[i]+n)
                choose=torch.rand(args.num_envs,1,device=dev)<beta
                obs,_,_,_,_=env.step(torch.where(choose,labels,pred))
        losses=[];biases=[]
        for _ in range(32):
            ids=torch.cat([v[torch.randint(len(v),(64,),device=dev)] for v in offline])
            x=[fixed_x[ids].float()];y=[fixed_y[ids]]
            available=[i for i in range(4) if counts[i]]
            for i in range(4):
                source=i if counts[i] else available[i%len(available)]
                ids=torch.randint(counts[source],(64,),device=dev)
                x.append(replay_x[source][ids].float());y.append(replay_y[source][ids])
            delta=student(torch.cat(x))-torch.cat(y)
            imitation=(delta.square()*weights).mean()
            bias=delta[:,12:14].mean(0).square().mean()
            loss=imitation+.5*bias
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step()
            losses.append(float(imitation.detach()));biases.append(float(bias.detach()))
        row=dict(iteration=iteration,teacher_mix=beta,weighted_loss=sum(losses)/len(losses),
            front_wheel_bias_loss=sum(biases)/len(biases),replay_counts=counts[:],seconds=round(time.perf_counter()-start,2))
        if (iteration+1)%25==0 or iteration==0:
            with torch.inference_mode():
                delta=student(fixed_x[held].float())-fixed_y[held]
                row.update(mujoco_heldout_mse=float(delta.square().mean()),
                    mujoco_heldout_front_wheel_rmse=float(delta[:,12:14].square().mean().sqrt()))
        print(json.dumps(row),flush=True);progress.write(json.dumps(row)+'\n');progress.flush()
        if (iteration+1)%50==0 or iteration==args.iterations-1:
            path=folder/f'student_{iteration}.pt';save(path,student,optimizer,'student',iteration,interface,.2)
            (folder/'latest.txt').write_text(str(path))
    progress.close()
    # Long-hold evaluation adds rolling drift and actual point slip to tilt/survival.
    env.cfg.randomize=False;env.cfg.episode_length_s=40.;results={}
    for name,model in [('teacher',teacher),('student',student.eval())]:
        torch.manual_seed(934179);obs,_=env.reset();alive=torch.ones(args.num_envs,dtype=torch.bool,device=dev)
        totals=dict(slip=0.,drift=0.,wheel_speed=0.,frames=0);good=torch.zeros(args.num_envs,device=dev)
        with torch.inference_mode():
            for k in range(1000):
                action=model(obs['policy' if name=='teacher' else 'student']);obs,_,term,trunc,_=env.step(action)
                alive&=~(term|trunc)
                if k>=250:
                    physical=env.physical();mask=alive
                    good+=(env.robot.data.projected_gravity_b.torch[:,0]>.9).float()
                    if bool(mask.any()):
                        totals['slip']+=float(physical['slip'][mask,:2].mean())
                        totals['drift']+=float(env.anchor_error(physical)[mask].norm(dim=-1).mean())
                        totals['wheel_speed']+=float(env.robot.data.joint_vel.torch[mask][:,env.wheel_ids[:2]].square().mean())
                        totals['frames']+=1
        n=max(totals.pop('frames'),1)
        results[name]=dict(survival_rate=float(alive.float().mean()),success_rate=float((alive&(good/750>.95)).float().mean()),
            front_point_slip_rms_m_s=(totals['slip']/n)**.5,front_anchor_drift_mean_m=totals['drift']/n,
            front_wheel_speed_rms_rad_s=(totals['wheel_speed']/n)**.5)
        (folder/'paired_evaluation20s.json').write_text(json.dumps(results,indent=2))
        print('[EVALUATION20S] '+json.dumps({name:results[name]}),flush=True)
    (folder/'exit_code').write_text('0')
except BaseException:
    traceback.print_exc();(folder/'exit_code').write_text('1');raise
finally:
    if env is not None:env.close()
    app.close()
