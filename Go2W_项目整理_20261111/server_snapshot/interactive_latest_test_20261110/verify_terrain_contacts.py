"""Finite runtime check of the actual test robot, standing policy, and terrain collision."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--log_dir',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
r=Path(__file__).resolve().parent;folder=Path(a.log_dir);folder.mkdir(parents=True,exist_ok=True)
cfg=json.loads((r/'config.json').read_text())
launcher=AppLauncher(a);app=launcher.app;env=None
try:
    import torch,omni.physx
    torch.set_num_threads(4)
    from scene import StandingEnv,EnvCfg,LEGS,WHEELS
    from terrain_setup import apply_terrain,inspect_loaded_terrain
    from control import Controller
    from plugin_adapters import load_adapter
    ec=EnvCfg();ec.sim.device=a.device or 'cuda:0'
    ec.sim.enable_scene_query_support=True
    ec.scene.robot.spawn.asset_path=str(Path(cfg['robot_asset_root'])/'urdf/go2w_description.urdf')
    initial=cfg['initial_standing'];ec.scene.robot.init_state.pos=tuple(initial['root_pose'][:3])
    ec.scene.robot.init_state.rot=tuple(initial['root_pose'][3:])
    ec.scene.robot.init_state.joint_pos=dict(zip(LEGS+WHEELS,initial['q']+initial['wheel_q']))
    ec.scene.robot.actuators['motors'].damping={'.*_hip_joint':0.,'.*_thigh_joint':0.,'.*_calf_joint':0.,'.*_foot_joint':100.}
    report=apply_terrain(ec,cfg,'skatepark')
    env=StandingEnv(ec,cfg);env.reset();report=inspect_loaded_terrain(report)
    c=Controller(cfg);c.contract=env.contract();c.policy_contracts['standing_hold']=env.contract('standing_hold')
    adapter=load_adapter(cfg['policies']['standing_hold'],r,env.device);adapter.bind_env(env)
    c.register('standing_hold',adapter);c._activate('standing_hold',env.snapshot())
    action=torch.zeros(1,16,device=env.device);heights=[];forces=[]
    with torch.inference_mode():
        for k in range(200):
            state=env.snapshot();c.tick(.02,state);state['command']=c.status()
            motor=c.execute(state,.02)
            if motor is not None:env.set_motor_targets(motor)
            env.step(action);c.after_step(env.snapshot())
            assert env.pd_updates==4 and torch.isfinite(env.robot.data.root_state_w.torch).all()
            heights.append(float(env.robot.data.root_pos_w.torch[0,2]))
            forces.append(float(env.contacts.data.net_forces_w_history.torch[:,:,:,2].amax(1).sum()))
    assert c.mode=='standing_hold',c.status()
    assert min(heights)>.30 and max(heights)<.5,(min(heights),max(heights))
    assert sum(forces[-50:])/50>100.,forces[-5:]
    query=omni.physx.get_physx_scene_query_interface();hits=[]
    for x,y,z in [(2.,2.,0.),(-14.,6.675,.03),(-14.,8.6,.18)]:
        hit=query.raycast_closest((x,y,3.),(0.,0.,-1.),6.)
        assert hit['hit'],(x,y,hit)
        point=list(hit['position']);assert abs(point[2]-z)<.005,(x,y,z,hit)
        assert str(hit['collision']).startswith('/World/TestTerrain/'),hit
        hits.append(dict(xy=[x,y],expected_z=z,hit_z=point[2],collision=str(hit['collision'])))
    env.reset();pos=env.robot.data.root_pos_w.torch[0].tolist()
    assert abs(pos[0])<1e-3 and abs(pos[1])<1e-3,pos
    report.update(status='passed',steps=200,policy=c.mode,standing_height_range=[min(heights),max(heights)],
        mean_support_force_n=sum(forces[-50:])/50,raycast_checks=hits,reset_root_xyz=pos)
    (folder/'validation.json').write_text(json.dumps(report,indent=2));print('[TERRAIN_VERIFIED] '+json.dumps(report),flush=True)
except BaseException:
    import traceback,sys
    traceback.print_exc();sys.stderr.flush()
    (folder/'failed').write_text(traceback.format_exc())
    raise
finally:
    if env:env.close()
    app.close()
