"""统一入口：先启动 Isaac App，再导入机器人/PhysX 模块。

train 训练教师；distill 让学生学习；play 验证；--zero 是零残差物理对照。
训练不会修改原有 MuJoCo 模型文件，导出也进入单独的输出目录。
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(mode):
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description="Go2-W Mimic teacher/student")
    parser.add_argument("--num_envs",type=int,default=1024 if mode!="play" else 4)
    parser.add_argument("--max_iterations",type=int,default=3000 if mode=="train" else 500)
    parser.add_argument("--checkpoint",help="resume teacher / play teacher or student / distillation teacher")
    parser.add_argument("--steps",type=int,default=800,help="play only; 0 = until window closed")
    parser.add_argument("--zero",action="store_true",help="play a physical zero-residual baseline")
    parser.add_argument("--randomized",action="store_true",help="enable randomization during evaluation")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log_dir")
    parser.add_argument("--real_time",action="store_true")
    parser.add_argument("--resume_reward_update",action="store_true",
                        help="train only: allow checked reward/termination changes when resuming")
    parser.add_argument("--resume_physics_update",action="store_true",
                        help="train only: explicit model/randomization correction; preserve policy interface")
    parser.add_argument("--video",action="store_true",help="record play-mode viewport to MP4")
    parser.add_argument("--video_length",type=int,default=400,help="play-mode recording length in env steps")
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if mode=="distill" and not args.checkpoint: parser.error("distill requires a teacher checkpoint")
    if mode=="play" and not args.zero and not args.checkpoint: parser.error("play requires --checkpoint or --zero")

    from go2w_mimic.settings import MOTION,URDF
    from go2w_mimic.interface import manifest,verify,write_json
    import torch
    checkpoint=None
    motion=MOTION
    if args.checkpoint:
        path=Path(args.checkpoint).expanduser().resolve()
        # 本脚本只加载自己训练/保存的 PyTorch checkpoint。
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        motion=path.parent/"params/stand_to_handstand.npz"
    interface=manifest(motion)
    if args.resume_reward_update and (mode!="train" or checkpoint is None):
        parser.error("--resume_reward_update requires train.py --checkpoint")
    if args.resume_physics_update and (mode!="train" or checkpoint is None):
        parser.error("--resume_physics_update requires train.py --checkpoint")
    if checkpoint: verify(checkpoint["interface"],interface,args.resume_reward_update,args.resume_physics_update)
    if mode in ("train","distill") and checkpoint and checkpoint["kind"]!="teacher":
        raise ValueError("This entry needs a teacher; student is evaluated via play.py")

    launcher=AppLauncher(args)
    app,env=launcher.app,None
    stamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder=Path(args.log_dir).expanduser().resolve() if args.log_dir else ROOT/"logs"/mode/stamp
    folder.mkdir(parents=True,exist_ok=True)
    try:
        from go2w_mimic.env import MimicEnv
        from go2w_mimic.env_cfg import MimicEnvCfg
        from go2w_mimic.learning import train_teacher,distill
        from go2w_mimic.networks import Teacher,Student
        torch.manual_seed(args.seed)
        cfg=MimicEnvCfg()
        cfg.scene.num_envs=args.num_envs
        cfg.sim.device=args.device or "cuda:0"
        cfg.seed=args.seed
        cfg.motion_path=str(motion)
        cfg.randomize=mode!="play" or args.randomized
        if checkpoint and mode=="train":
            cfg.domain_level=checkpoint.get("curriculum",cfg.domain_level)
        cfg.viewer.eye=(2.,2.,1.2)
        cfg.viewer.lookat=(0.,0.,.5)
        if args.video:
            if mode!="play":
                raise ValueError("--video is only supported by play.py")
            from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
            cfg.video_recorders=[VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=str(folder/"videos"),
                video_length=args.video_length,
                output_filename_prefix="policy",
            )]
        env=MimicEnv(cfg)
        params=folder/"params"
        params.mkdir(exist_ok=True)
        motion_copy=params/"stand_to_handstand.npz"
        if Path(motion).resolve()!=motion_copy.resolve():
            shutil.copy2(motion,motion_copy)
        write_json(params/"interface.json",interface)
        write_json(params/"arguments.json",vars(args))
        if checkpoint:
            if checkpoint.get("migration"):
                write_json(params/"checkpoint_migration.json",checkpoint["migration"])
            write_json(params/"resume_changes.json",dict(parent=str(path),
                parent_iteration=checkpoint["iteration"],
                changed={k:dict(before=checkpoint["interface"].get(k),after=v)
                    for k,v in interface.items() if checkpoint["interface"].get(k)!=v}))
        # 运行时导入后的实际质量/惯量才是跨仿真校准依据。
        write_json(params/"physics_model.json",dict(body_names=env.robot.body_names,
            joint_names=env.robot.joint_names,nominal_mass_kg=env.nominal_masses[0].tolist(),
            nominal_inertia_kg_m2=env.nominal_inertias[0].tolist(),
            leg_soft_limits_rad=env.limits[0].cpu().tolist(),
            note="Nominal PhysX tensors before per-environment mass scaling. Compare coordinate frames."))
        d=env.robot.data
        write_json(params/"physics_runtime.json",dict(
            body_names=env.robot.body_names,joint_names=env.robot.joint_names,
            total_nominal_mass_kg=float(env.nominal_masses[0].sum()),
            nominal_mass_kg=env.nominal_masses[0].tolist(),
            nominal_inertia_kg_m2=env.nominal_inertias[0].tolist(),
            com_pose_in_link_xyzw=d.body_com_pose_b.torch[0].cpu().tolist(),
            joints={name:getattr(d,name).torch[0].cpu().tolist() for name in (
                "joint_pos_limits","joint_vel_limits","joint_effort_limits",
                "joint_stiffness","joint_damping","joint_armature","joint_friction_coeff",
                "joint_dynamic_friction_coeff","joint_viscous_friction_coeff")},
            domain_level=env.domain_level,
            mass_scale_range=[float(env.mass_scale.min()),float(env.mass_scale.max())],
            robot_friction_range=[float(env.mu.min()),float(env.mu.max())],
            physics_dt=env.physics_dt,control_dt=env.step_dt,
            enabled_self_collisions=cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
            note="PhysX COM pose xyzw relative to link; inertia is backend tensor, interpret with COM orientation. Ground mu=.8, average combine."))
        shutil.copytree(ROOT/"go2w_mimic",params/"code/go2w_mimic",
                        ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        shutil.copytree(ROOT/"scripts",params/"code/scripts",
                        ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        print(f"[Go2W] mode={mode} output={folder}",flush=True)
        if mode=="train":
            train_teacher(env,app,folder,interface,args.max_iterations,checkpoint)
        elif mode=="distill":
            distill(env,app,folder,interface,args.max_iterations,checkpoint)
        else:
            model=None
            if checkpoint:
                model=(Teacher() if checkpoint["kind"]=="teacher" else Student()).to(env.device)
                model.load_state_dict(checkpoint["model"]); model.eval()
            obs,_=env.reset()
            traces=open(folder/"trace.jsonl","w",encoding="utf-8")
            episodes=[]
            last_episode=None
            try:
                with torch.no_grad():
                    step=0
                    while app.is_running() and (args.steps==0 or step<args.steps):
                        tick=time.perf_counter()
                        action=(torch.zeros(env.num_envs,16,device=env.device) if model is None
                            else model(obs["policy" if checkpoint["kind"]=="teacher" else "student"]))
                        obs,_,term,trunc,info=env.step(action)
                        p=env.physical()
                        row=dict(step=step,episode_step=int(env.episode_length_buf[0]),
                            height=float(env.robot.data.root_pos_w.torch[0,2]-env.scene.env_origins[0,2]),
                            gravity=env.robot.data.projected_gravity_b.torch[0].tolist(),
                            contact=p["contact"][0].tolist(),wheel_speed=env.robot.data.joint_vel.torch[0,env.wheel_ids].tolist(),
                            leg_target=env.target[0].tolist(),leg_actual=env.robot.data.joint_pos.torch[0,env.leg_ids].tolist(),
                            wheel_target=env.wheel_target[0].tolist(),torque=env.torque[0].tolist(),
                            hold=float(env.hold[0]),front_force=p["force"][0,:,:2].mean(0).tolist(),
                            reset_this_step=bool((term|trunc)[0]))
                        traces.write(json.dumps(row)+"\n")
                        if step%50==0: print(json.dumps(row),flush=True)
                        if (term|trunc).any() and "episode" in info:
                            episodes.append(info["episode"])
                        step+=1
                        if args.real_time: time.sleep(max(0,env.step_dt-(time.perf_counter()-tick)))
            finally:
                traces.close()
                write_json(folder/"evaluation.json",dict(episodes=episodes,
                    note="Only completed full episodes count. Trace rows flagged reset are new-episode states."))
    except KeyboardInterrupt:
        print("[Go2W] stopped; training entry saves last checkpoint in finally",flush=True)
    finally:
        if env is not None: env.close()
        app.close()


if __name__=="__main__":
    run("train")
