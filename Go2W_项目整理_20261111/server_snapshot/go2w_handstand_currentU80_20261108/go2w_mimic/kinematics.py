"""只用 NumPy 读取 URDF 并计算正向运动学，离线生成参考时无需 Isaac。

FK 回答“给定关节角，各部件在哪里”。它不计算接触力，也不能证明动作可执行。
轮的位置定义为 foot link 原点（轮轴）；轮胎几何有横向偏移，接触验证另做。
"""
import xml.etree.ElementTree as ET
import numpy as np


def vec(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    r, p, y = rpy
    T[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    T[:3, 3] = xyz
    return T


class Kinematics:
    def __init__(self, path):
        xml = ET.parse(path).getroot()
        self.joints, self.limits, self.mass = [], {}, {}
        children = set()
        for j in xml.findall("joint"):
            o = j.find("origin")
            T = transform(vec(o.get("xyz") if o is not None else None),
                          vec(o.get("rpy") if o is not None else None))
            child, parent = j.find("child").get("link"), j.find("parent").get("link")
            axis = j.find("axis")
            self.joints.append((j.get("name"), parent, child, T, j.get("type"),
                                vec(axis.get("xyz") if axis is not None else "1 0 0")))
            children.add(child)
            lim = j.find("limit")
            if j.get("type") == "revolute":
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        self.root = next(x.get("name") for x in xml.findall("link") if x.get("name") not in children)
        for link in xml.findall("link"):
            i = link.find("inertial")
            if i is not None:
                o = i.find("origin")
                self.mass[link.get("name")] = (
                    float(i.find("mass").get("value")), vec(o.get("xyz") if o is not None else None))

    def fk(self, angles, root=None):
        result = {self.root: np.eye(4) if root is None else root}
        todo = self.joints[:]
        while todo:
            remaining = []
            for name, parent, child, origin, kind, axis in todo:
                if parent not in result:
                    remaining.append((name, parent, child, origin, kind, axis))
                    continue
                motion = np.eye(4)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = rotation(axis, angles.get(name, 0.0))
                result[child] = result[parent] @ origin @ motion
            if len(remaining) == len(todo):
                raise ValueError("URDF joint graph is disconnected")
            todo = remaining
        return result

    def com(self, poses):
        total = sum(m for m, _ in self.mass.values())
        return sum(m*(poses[name][:3, :3]@p + poses[name][:3, 3])
                   for name, (m, p) in self.mass.items())/total

    def solve_leg(self, leg, target_b, initial):
        """阻尼最小二乘 IK：轮轴目标在机身坐标系，返回该腿 3 个角度。

        数值 Jacobian 避免把某台机器人的关节轴方向写死在解析公式里。
        """
        names = [f"{leg}_{p}_joint" for p in ("hip", "thigh", "calf")]
        limits = np.array([self.limits[n] for n in names])
        lo, hi = limits.mean(1)-np.diff(limits)[:, 0]*.475, limits.mean(1)+np.diff(limits)[:, 0]*.475
        q = np.array(initial, float)
        def position(v):
            return self.fk(dict(zip(names, v)))[f"{leg}_foot"][:3, 3]
        for _ in range(60):
            p = position(q)
            e = np.asarray(target_b)-p
            if np.linalg.norm(e) < 1e-6:
                return q
            J = np.column_stack([(position(q+np.eye(3)[k]*1e-5)-p)/1e-5 for k in range(3)])
            dq = J.T @ np.linalg.solve(J@J.T + np.eye(3)*1e-7, e)
            q = np.clip(q+np.clip(dq, -.15, .15), lo, hi)
        if np.linalg.norm(position(q)-target_b) > .002:
            raise ValueError(f"{leg} IK target unreachable: {target_b}")
        return q
