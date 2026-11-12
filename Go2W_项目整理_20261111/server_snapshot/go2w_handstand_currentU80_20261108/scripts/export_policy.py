"""导出学生 + 同版本参考/URDF/接口，供后续 MuJoCo 集成；不替换现有 ONNX。

本项目学生输入 2555 维，旧 sim.py 的 103 维接口不能直接加载。
必须按 interface.json 增加历史缓冲和相同参考查表/控制器，然后再验收。
"""
import argparse
import shutil
import sys
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from go2w_mimic.interface import manifest,verify,write_json,sha
from go2w_mimic.networks import Student
from go2w_mimic.settings import STUDENT_DIM


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    checkpoint=Path(a.checkpoint).expanduser().resolve()
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["kind"]!="student":
        raise ValueError("Teacher uses privileged state. Distill a student before deployment export.")
    motion=checkpoint.parent/"params/stand_to_handstand.npz"
    verify(saved["interface"],manifest(motion))
    model=Student().eval()
    model.load_state_dict(saved["model"])
    out=Path(a.output).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    path=out/"policy.onnx"
    torch.onnx.export(model,torch.zeros(1,STUDENT_DIM),str(path),opset_version=17,dynamo=False,
        input_names=["observation"],output_names=["bounded_action"],
        dynamic_axes={"observation":{0:"batch"},"bounded_action":{0:"batch"}})
    import onnxruntime as ort
    runtime=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
    x=np.random.default_rng(42).normal(0,.1,(8,STUDENT_DIM)).astype("f4")
    with torch.no_grad(): expected=model(torch.from_numpy(x)).numpy()
    actual=runtime.run(None,{"observation":x})[0]
    np.testing.assert_allclose(expected,actual,atol=2e-5,rtol=1e-4)
    shutil.copy2(motion,out/"stand_to_handstand.npz")
    shutil.copy2(checkpoint,out/checkpoint.name)
    shutil.copytree(ROOT/"assets/go2w_description",out/"go2w_description",dirs_exist_ok=True)
    shutil.copy2(checkpoint.parent/"params/physics_model.json",out/"physics_model.json")
    write_json(out/"interface.json",saved["interface"])
    write_json(out/"export_report.json",dict(checkpoint=str(checkpoint),onnx_sha256=sha(path),
        max_abs_error=float(abs(expected-actual).max()),
        scope="Numerical inference equivalence only. No MuJoCo or hardware success claim."))
    print(out)


if __name__=="__main__":
    main()
