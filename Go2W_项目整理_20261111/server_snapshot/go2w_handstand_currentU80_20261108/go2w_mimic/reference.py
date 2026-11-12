"""运行时只查表，不运行 IK，不把机器人瞬移到参考位置。

phase_time 是独立、单调前进的参考时钟；每回合随机播放速度。
它不由当前 pitch 倒推出进度，因而能给出“下一步要做什么”的目标。
末帧进入保持，不循环回到站立。学生可在部署端重建同一张参考表。
"""
import numpy as np
import torch


class Motion:
    def __init__(self, path, device="cpu"):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {k: torch.as_tensor(archive[k].copy(), dtype=torch.float32, device=device)
                         for k in archive.files}
        self.duration = float(self.data["time"][-1])
        self.dt = float(self.data["time"][1]-self.data["time"][0])

    def sample(self, time):
        index = (time.clamp(0, self.duration)/self.dt).clamp(max=len(self.data["time"])-1)
        lo = index.floor().long()
        hi = (lo+1).clamp(max=len(self.data["time"])-1)
        alpha = index-lo
        out = {}
        for name, table in self.data.items():
            if name == "time":
                continue
            w = alpha.reshape((-1,) + (1,)*(table.ndim-1))
            out[name] = table[lo]*(1-w)+table[hi]*w
        # 四元数此处不用于奖励；gravity 的线性插值后归一化避免长度变化。
        out["gravity"] = torch.nn.functional.normalize(out["gravity"], dim=-1)
        return out
