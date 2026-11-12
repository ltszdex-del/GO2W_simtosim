"""Independent H warm-up -> return-PPO replay, using the original training environment."""
import argparse,hashlib,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--steps',type=int,default=0)
parser.add_argument('--log_dir',required=True)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--nominal',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
env=None
try:
    import torch
    from env import ReturnCfg,ReturnEnv
    from model import Policy
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    checkpoint=Path(args.checkpoint).resolve()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=ck['manifest']
    assert ck['kind']=='handstand_return' and m['schema']=='go2w-handstand-return-v1'
    for name,digest in m['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Training source mismatch: '+name
    for name,digest in m['state_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Initial/goal state mismatch: '+name
    assert hashlib.sha256((ROOT/'teacher_4099.pt').read_bytes()).hexdigest()==m['teacher']['H_sha256']
    assert hashlib.sha256((ROOT/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()).hexdigest()==m['urdf_sha256']
    cfg=ReturnCfg();cfg.scene.num_envs=1;cfg.seed=args.seed;cfg.sim.device=args.device or 'cuda:0'
    cfg.physics_randomization=not args.nominal
    cfg.viewer.eye=(2.0,2.0,1.25);cfg.viewer.lookat=(0.,0.,.40)
    env=ReturnEnv(cfg);env.level=ck.get('curriculum',0.)
    actor=Policy().to(env.device).eval();actor.load_state_dict(ck['model'],strict=True)
    obs,_=env.reset()
    import carb.settings
    settings=carb.settings.get_settings()
    settings.set('/rtx/post/aa/op',4);settings.set('/rtx/post/dlss/execMode',2)
    folder=Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    metadata=dict(checkpoint=str(checkpoint),checkpoint_iteration=ck['iteration'],completed_updates=ck['iteration']+1,
        physics_randomization=not args.nominal,seed=args.seed,num_envs=1,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        playback='deterministic actor; real H4099 warm-up before each return attempt; original failure/auto-reset rules',
        camera='initial framing only; free camera navigation',training_modified=False)
    (folder/'replay_manifest.json').write_text(json.dumps(metadata,indent=2))
    label=None
    try:
        import omni.ui as ui
        window=ui.Window('Handstand return replay',width=460,height=150)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label(f"Return PPO: {ck['iteration']+1} updates | frozen H4099 warm-up")
                label=ui.Label('Starting replay...')
                ui.Label('Automatic repeats after success/failure. Camera is free.')
                ui.Label('Training continues in a separate process.')
    except Exception as error:print('[UI]',repr(error),flush=True)
    print('[READY] RETURN_REPLAY',json.dumps(metadata),flush=True)
    step=completed=successes=0;last_phase=None;started=time.monotonic()
    with torch.inference_mode(),(folder/'trace.jsonl').open('w',buffering=1) as trace:
        while app.is_running() and (args.steps==0 or step<args.steps):
            tick=time.monotonic()
            phase='RETURN PPO' if bool(env.recovering[0]) else 'H4099 WARM-UP'
            if phase!=last_phase:
                print('[PHASE]',phase,'step',step,flush=True);last_phase=phase
            obs,reward,term,trunc,info=env.step(actor(obs['policy']))
            if not torch.isfinite(obs['policy']).all() or not torch.isfinite(env.torque).all():
                raise RuntimeError('Nonfinite replay state')
            assert env.pd_updates==4
            if 'episode' in info:
                episode=dict(info['episode']);completed+=episode['count']
                successes+=round(episode['success']*episode['count'])
                print('[EPISODE]',json.dumps(episode),flush=True)
            step+=1
            if step%10==0:
                d=env.robot.data;g=d.projected_gravity_b.torch[0]
                status=dict(**metadata,step=step,sim_s=step*.02,wall_s=round(time.monotonic()-started,2),
                    phase='RETURN PPO' if bool(env.recovering[0]) else 'H4099 WARM-UP',
                    completed_episodes=completed,successful_episodes=successes,
                    height_m=float(d.root_pos_w.torch[0,2]),pitch_deg=float(torch.atan2(g[0],-g[2])*180/math.pi),
                    rear_landed=(env.landing_time[0]>=0).tolist(),hold_s=float(env.hold[0]),
                    peak_torque_nm=float(env.torque.abs().max()),pd_substeps=env.pd_updates,
                    root_position=d.root_pos_w.torch[0].tolist())
                (folder/'status.json').write_text(json.dumps(status,indent=2));trace.write(json.dumps(status)+'\n')
                if label:label.text=f"{status['phase']} | pitch {status['pitch_deg']:.1f} deg | episodes {completed}, success {successes}"
            if args.livestream:time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
except BaseException:
    traceback.print_exc()
    if env:env.close()
    app.close(exit_code=1)
else:
    if env:env.close()
    app.close()
