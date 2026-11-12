"""用 MuJoCo 碰撞检测审核整个参考；仅 mj_forward，不训练或积分。

已有本地 .venv-mujoco 可运行。本脚本核对 URDF FK 与 MJCF 的轮轴位置，
并报告穿地、自碰撞。它不把“几何检查通过”写成“倒立动态成功”。
"""
import json
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]


def main():
    model=mujoco.MjModel.from_xml_path(str(ROOT/"assets/scene_training_urdf.xml"))
    data=mujoco.MjData(model)
    ref=np.load(ROOT/"motions/stand_to_handstand.npz",allow_pickle=False)
    names=[f"{l}_{p}_joint" for l in ("FR","FL","RR","RL") for p in ("hip","thigh","calf")]
    wheels=[model.body(f"{l}_wheel_link").id for l in ("FR","FL","RR","RL")]
    floor=model.geom("floor").id
    violations=[]
    max_fk_error=0.
    for i,t in enumerate(ref["time"]):
        data.qpos[:]=model.qpos0
        pitch=np.arctan2(ref["gravity"][i,0],-ref["gravity"][i,2])
        data.qpos[:3]=ref["root_position"][i]
        data.qpos[3:7]=[np.cos(pitch/2),0,np.sin(pitch/2),0]
        for name,q in zip(names,ref["joints"][i]):
            data.qpos[model.joint(name).qposadr]=q
        mujoco.mj_forward(model,data)
        rotation=data.xmat[model.body("base_link").id].reshape(3,3)
        actual=(data.xpos[wheels]-data.qpos[:3])@rotation
        max_fk_error=max(max_fk_error,float(abs(actual-ref["feet_body"][i]).max()))
        for c in data.contact:
            if c.dist < -.002:
                violations.append(dict(time=float(t),depth=float(-c.dist),
                    bodies=[model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)],
                    ground=bool(floor in (c.geom1,c.geom2))))
    report=dict(max_axle_fk_error_m=max_fk_error,penetrations_deeper_than_2mm=violations,
        passed=max_fk_error<1e-4 and not violations,
        scope="Full reference kinematics and MuJoCo collision geometry only; no dynamics rollout.")
    (ROOT/"motions/mujoco_geometry_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__":
    main()
