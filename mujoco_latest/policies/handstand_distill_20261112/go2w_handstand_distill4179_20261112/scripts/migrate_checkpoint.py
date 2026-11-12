"""把旧观测维度的 teacher checkpoint 迁移到新观测维度。

方案：观测末尾新增了 yaw sin/cos 两维（追加在 proprio 末尾），
Teacher 的 actor/critic 第一层 Linear 输入维从旧 dim 变到新 dim。
迁移做法：把新增两列权重置零、其余列原样保留，保证迁移后网络
对旧输入产生完全相同的输出（新维初值贡献为 0），实现无损续训。

用法（在包含新 settings.py 的代码目录下运行）:
    python scripts/migrate_checkpoint.py <src.pt> <dst.pt>
"""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2w_mimic.settings import TEACHER_DIM, GOAL_DIM, PRIV_DIM, MOTION
from go2w_mimic.interface import manifest

HEADING_DIM = 2  # 新增 yaw sin/cos 维数，追加在 proprio 末尾


def main():
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    old_dim = ckpt["model"]["actor.0.weight"].shape[1]
    new_dim = TEACHER_DIM
    assert new_dim == old_dim + HEADING_DIM, \
        f"维度不一致: old={old_dim}, new={new_dim}（期望相差 {HEADING_DIM}）"

    # 旧 proprio 长度 = 旧总 dim - GOAL_DIM - PRIV_DIM，即 heading 的插入列位置。
    insert_at = old_dim - GOAL_DIM - PRIV_DIM

    for key in ("actor.0.weight", "critic.0.weight"):
        w = ckpt["model"][key]
        assert w.shape[1] == old_dim, f"{key} 列数 {w.shape[1]} != 旧维度 {old_dim}"
        nw = torch.zeros(w.shape[0], new_dim, dtype=w.dtype)
        nw[:, :insert_at] = w[:, :insert_at]
        nw[:, insert_at + HEADING_DIM:] = w[:, insert_at:]
        ckpt["model"][key] = nw

    # 同步迁移 Adam 优化器状态：第一层权重的 exp_avg / exp_avg_sq 形状也变了，
    # 否则 train_teacher 里 optimizer.load_state_dict 会因维度不匹配崩溃。
    # 遍历状态里所有键，按「2D 且列数 == 旧维度」定位第一层权重矩（键名不硬编码）。
    opt_state = ckpt["optimizer"]["state"]
    for idx, st in opt_state.items():
        for moment in list(st.keys()):
            m = st[moment]
            if isinstance(m, torch.Tensor) and m.ndim == 2 and m.shape[1] == old_dim:
                nm = torch.zeros(m.shape[0], new_dim, dtype=m.dtype)
                nm[:, :insert_at] = m[:, :insert_at]
                nm[:, insert_at + HEADING_DIM:] = m[:, insert_at:]
                st[moment] = nm

    # 用新代码重算接口契约（新维度、新 sources、新 reward_weights），
    # 使后续 train.py 的 verify() 直接通过，无需 --resume_reward_update。
    motion = src.parent / "params" / "stand_to_handstand.npz"
    if not motion.exists():
        motion = MOTION
    ckpt["interface"] = manifest(motion)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"[migrate] {src.name} -> {dst.name}")
    print(f"[migrate] 维度 {old_dim} -> {new_dim}，heading 插入列 {insert_at}-{insert_at + HEADING_DIM - 1}")
    print(f"[migrate] motion={motion}")


if __name__ == "__main__":
    main()
