"""Latest Go2W policies in local MuJoCo. See README.md for validation limits."""
import bootstrap
import argparse
import ctypes
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import mujoco
from backend import Environment
from control import Controller
from plugin_adapters import load_adapter

torch.set_num_threads(1)
if hasattr(__import__('sys').stdout,'reconfigure'):__import__('sys').stdout.reconfigure(encoding='utf-8')

def create_runtime():
    cfg=bootstrap.load_config()
    scene=bootstrap.ROOT/'generated/scene.xml'
    if not scene.exists():
        from build_scene import build
        scene=build(cfg)
    env=Environment(cfg,scene)
    controller=Controller(cfg)
    controller.contract=env.contract()
    for mode in cfg['policies']:
        controller.policy_contracts[mode]=env.contract(mode)
        try:
            adapter=load_adapter(cfg['policies'][mode],bootstrap.RUNTIME,'cpu')
            controller.register(mode,adapter);adapter.bind_env(env)
            print('[LOADED]',mode,Path(adapter.path).name,flush=True)
        except Exception as error:
            controller.adapters.pop(mode,None)
            controller.policy_errors[mode]=repr(error)
            traceback.print_exc()
    if len(controller.adapters)!=len(cfg['policies']):
        raise RuntimeError('Some latest policies failed validation: '+str(controller.policy_errors))
    return env,controller


def advance(env,controller):
    if controller.reset_requested:
        env.reset();controller.reset_runtime(env.snapshot())
    state=env.snapshot();controller.tick(.02,state)
    if controller.pending=='handstand' and controller.active is None:
        env.set_motor_targets(controller.adapters['handstand'].preparation_targets())
    state['command']=controller.status()
    motor=controller.execute(state,.02)
    if motor is not None:env.set_motor_targets(motor)
    env.step()
    live=env.snapshot();completed=controller.after_step(live)
    if completed=='standing':env.standing_target()
    if completed=='error':env.safe_stop()
    if controller.mode in ('standing','locomotion','standing_hold'):
        if live['root_position_w'][0,2]<.18 or live['projected_gravity_b'][0,2]>-.5:
            controller.mode='error';controller.message='Robot fell in MuJoCo; press R to reset'
            controller.active=None;env.safe_stop();controller.release(controller.source)
    controller.state=live
    controller.physics=dict(engine='MuJoCo '+mujoco.__version__,sim_s=round(env.data.time,3),
        xyz=live['root_position_w'][0].tolist(),gravity_b=live['projected_gravity_b'][0].tolist(),
        peak_torque_nm=float(np.max(np.abs(env.data.qfrc_actuator[env.dids]))),pd_substeps=env.pd_updates,
        contacts=env.data.ncon,motor_dt_s=env.step_dt/env.pd_updates)
    return controller.status()


def draw(viewer,env,status):
    scene=viewer.user_scn;scene.ngeom=0
    center=env.data.subtree_com[1].copy();yaw=env.snapshot()['yaw_rad']
    ref=status['command_m_s_rad_s'];actual=status['actual_velocity_yaw_m_s_rad_s']
    for values,color,radius in ((ref,[.1,.9,.2,1],.012),(actual,[.2,.5,1.,1],.008)):
        vx,vy,wz=values;end=center+np.array([np.cos(yaw)*vx-np.sin(yaw)*vy,np.sin(yaw)*vx+np.cos(yaw)*vy,0])*.3
        for endpoint in (end,center+np.array([0,0,wz*.16])):
            if np.linalg.norm(endpoint-center)<1e-5:continue
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,radius,center,endpoint);scene.ngeom+=1
    viewer.cam.lookat[:]=env.data.xpos[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--scenario',default='idle',choices=['idle','W','S','A','D','Q','E','WQ','WE','SQ','SE','P','U','H','HH'])
    parser.add_argument('--wheel',action='store_true')
    parser.add_argument('--low',action='store_true',help='Same K half-limit mode as the Isaac test environment')
    parser.add_argument('--log',default='live')
    args=parser.parse_args()
    folder=bootstrap.ROOT/'logs'/args.log;folder.mkdir(parents=True,exist_ok=True)
    env,controller=create_runtime()
    (folder/'config.json').write_text(json.dumps(env.control,indent=2),encoding='utf-8')
    (folder/'contracts.json').write_text(json.dumps(controller.policy_contracts,indent=2),encoding='utf-8')
    if args.wheel:controller._set_mobility('wheel',env.snapshot())
    if args.low:
        controller.feed('validation',['K']);controller.tick(.02,env.snapshot());controller.feed('validation',[])
    viewer=None;server=None
    if not args.headless:
        import mujoco.viewer
        from bridge import serve
        server=serve(controller)
        from plain_viewer import Viewer
        viewer=Viewer(env.model,env.data)
        viewer.cam.distance=2.6;viewer.cam.azimuth=135;viewer.cam.elevation=-20
        viewer.opt.geomgroup[3]=0
        print('[READY] http://127.0.0.1:8776  WASDQE / M / K / H / P / U / R',flush=True)
    step=0;start=time.monotonic();summaries=[]
    try:
        with (folder/'trace.jsonl').open('w',encoding='utf-8') as trace,torch.inference_mode():
            while (not args.steps or step<args.steps) and (viewer is None or viewer.is_running()):
                tick=time.monotonic()
                if args.headless:
                    keys=list(args.scenario) if args.scenario!='idle' and 100<=step<250 else []
                    if args.scenario in ('H','HH','P','U'):
                        keys=[args.scenario[0]] if step==100 else []
                        if args.scenario=='HH' and step==400:keys=['H']
                        if args.scenario=='U':
                            keys=['P'] if step==100 else (['U'] if step==450 else [])
                    controller.feed('validation',keys)
                elif os.name=='nt':
                    user32=ctypes.windll.user32
                    foreground=user32.GetForegroundWindow();pid=ctypes.c_ulong()
                    user32.GetWindowThreadProcessId(foreground,ctypes.byref(pid))
                    if pid.value==os.getpid():
                        keys=[k for k in 'WASDQEMKHPUR' if user32.GetAsyncKeyState(ord(k))&0x8000]
                        controller.feed('webrtc-keyboard',keys,confirmed_held=True)
                    elif controller.source=='webrtc-keyboard':controller.release('webrtc-keyboard')
                status=advance(env,controller);step+=1
                if step%10==0:
                    trace.write(json.dumps(status,ensure_ascii=False)+'\n');trace.flush()
                    (folder/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
                    summaries.append(dict(step=step,mode=status['mode'],actual=status['actual_velocity_yaw_m_s_rad_s'],
                        command=status['command_m_s_rad_s'],height=status['physics']['xyz'][2],gravity=status['physics']['gravity_b'],torque=status['physics']['peak_torque_nm']))
                if viewer:
                    with viewer.lock():draw(viewer,env,status)
                    limits=status['command_limits'];ref=status['command_m_s_rad_s'];actual=status['actual_velocity_yaw_m_s_rad_s']
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        'Go2W | MuJoCo sim-to-sim\nMode / speed\nRef vx vy wz\nReal vx vy wz\nLimits vx / wz\nControls\nActions\nArrows',
                        f"latest policies\n{status['mobility_profile']} / {status['speed_mode']} / {status['mode']}\n"
                        f'{ref[0]:+.2f} {ref[1]:+.2f} {ref[2]:+.2f}\n{actual[0]:+.2f} {actual[1]:+.2f} {actual[2]:+.2f}\n'
                        f"{limits['max_forward_m_s']:.2f} m/s / {limits['max_yaw_rad_s']:.2f} rad/s\nWASDQE | M drive | K half limits\nH handstand/return | P down | U up | R reset\ngreen=reference blue=actual; vertical=yaw"))
                    viewer.sync();time.sleep(max(0,.02-(time.monotonic()-tick)))
                if controller.mode=='error' and args.headless:break
    finally:
        if viewer:viewer.close()
        if server:server.shutdown()
        result=dict(scenario=args.scenario,wheel=args.wheel,steps=step,sim_s=env.data.time,wall_s=time.monotonic()-start,
                    status=controller.status(),samples=summaries)
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[RESULT]',args.scenario,'wheel='+str(args.wheel),'steps='+str(step),'mode='+controller.mode,flush=True)

if __name__=='__main__':main()
