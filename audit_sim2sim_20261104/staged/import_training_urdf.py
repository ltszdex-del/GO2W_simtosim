"""Build a MuJoCo robot from the Go2W URDF used by robot_lab.

MuJoCo's URDF importer fixes the root to the world and creates no actuators.
We first let it resolve fixed joints and collision primitives, then restore a
free root, URDF base inertia, and torque actuators in the exported MJCF.
Visual DAE files are omitted because MuJoCo's native URDF importer does not
load them; the resulting robot renders its collision geometry.
"""

from __future__ import annotations

import argparse
import copy
import os
import math
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_URDF = ROOT / "source" / "go2w_description.urdf"
DEFAULT_OUTPUT = ROOT / "go2w_training.xml"


def rotation(rpy: str) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]]
    )


def origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.eye(3), np.zeros(3)
    return rotation(element.get("rpy", "0 0 0")), np.fromstring(element.get("xyz", "0 0 0"), sep=" ")


def root_inertia(urdf: ET.Element, subtree: str | None = None) -> dict[str, str]:
    """Combine the base and its fixed descendants, such as the two head links."""
    links = {link.get("name"): link for link in urdf.findall("link")}
    child_names = {joint.find("child").get("link") for joint in urdf.findall("joint")}
    root_name = subtree or next(name for name in links if name not in child_names)
    fixed = {}
    for joint in urdf.findall("joint"):
        if joint.get("type") == "fixed":
            fixed.setdefault(joint.find("parent").get("link"), []).append(joint)

    mass_sum = 0.0
    first_moment = np.zeros(3)
    inertia_origin = np.zeros((3, 3))
    stack = [(root_name, np.eye(3), np.zeros(3))]
    while stack:
        name, parent_r, parent_p = stack.pop()
        inertial = links[name].find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            local_r, local_p = origin(inertial.find("origin"))
            at = parent_p + parent_r @ local_p
            axis = parent_r @ local_r
            i = inertial.find("inertia")
            tensor = np.array(
                [[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                 [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                 [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]]
            )
            mass_sum += mass
            first_moment += mass * at
            inertia_origin += axis @ tensor @ axis.T + mass * ((at @ at) * np.eye(3) - np.outer(at, at))
        for joint in fixed.get(name, []):
            local_r, local_p = origin(joint.find("origin"))
            stack.append((joint.find("child").get("link"), parent_r @ local_r, parent_p + parent_r @ local_p))
    center = first_moment / mass_sum
    tensor = inertia_origin - mass_sum * ((center @ center) * np.eye(3) - np.outer(center, center))
    fmt = lambda values: " ".join(f"{float(value):.12g}" for value in values)
    return {
        "pos": fmt(center), "mass": f"{mass_sum:.12g}",
        "fullinertia": fmt([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                            tensor[0, 1], tensor[0, 2], tensor[1, 2]]),
    }


def build(urdf_path: Path = DEFAULT_URDF, output_path: Path = DEFAULT_OUTPUT) -> None:
    urdf = ET.parse(urdf_path).getroot()
    base_inertia = root_inertia(urdf)
    # Collision shapes in the source URDF are primitives. DAE visuals are not
    # needed to obtain the training asset's kinematics and contact geometry.
    for link in urdf.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    with tempfile.TemporaryDirectory() as temporary:
        stripped = Path(temporary) / "collision_only.urdf"
        imported = Path(temporary) / "imported.xml"
        ET.ElementTree(urdf).write(stripped, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(stripped))
        mujoco.mj_saveLastXML(str(imported), model)
        mjcf = ET.parse(imported).getroot()

    world = mjcf.find("worldbody")
    if world is None:
        raise RuntimeError("MuJoCo URDF import did not produce a worldbody")
    base = ET.Element("body", {"name": "base_link"})
    ET.SubElement(base, "freejoint", {"name": "root"})
    ET.SubElement(base, "inertial", base_inertia)
    for child in list(world):
        world.remove(child)
        base.append(child)
    world.append(base)

    # Recompute ALL merged inertias from explicit URDF inertials. MuJoCo's
    # importer otherwise assigns density-derived mass to collision-only fixed
    # links (calflower), unlike the saved Isaac nominal model.
    for body in mjcf.iter("body"):
        source = "base" if body.get("name") == "base_link" else body.get("name")
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)
        body.insert(0, ET.Element("inertial", root_inertia(urdf, source)))
    ET.SubElement(mjcf, "option", {"timestep": "0.005", "cone": "elliptic"})

    # Names match the existing sim.py interface; only names change, not frames.
    for body in mjcf.iter("body"):
        name = body.get("name", "")
        if name.endswith("_foot"):
            body.set("name", name.replace("_foot", "_wheel_link"))
    for joint in mjcf.iter("joint"):
        name = joint.get("name", "")
        if name.endswith("_foot_joint"):
            joint.set("name", name.replace("_foot_joint", "_wheel_joint"))
    for geom in mjcf.iter("geom"):
        # PhysX nominal material has sliding friction, without an explicit
        # rolling/torsional resistance model. Avoid condim=6 wheel braking.
        if geom.get("type") == "cylinder" and geom.get("size", "").startswith("0.086 "):
            geom.set("condim", "3")
            geom.set("friction", "0.8 0 0")

    # Reuse the original MJCF render meshes only. Inertias and collision
    # primitives above remain sourced from the training URDF.
    visual_path = urdf_path.parent.parent / "go2w.xml"
    visual = ET.parse(visual_path).getroot()
    mjcf.find("compiler").set("meshdir", os.path.relpath(visual_path.parent / "assets", output_path.parent).replace("\\", "/"))
    mjcf.append(copy.deepcopy(visual.find("asset")))
    for geom in mjcf.iter("geom"):
        geom.set("group", "3")
    bodies = {body.get("name"): body for body in mjcf.iter("body")}
    for body in visual.iter("body"):
        target = bodies.get(body.get("name"))
        if target is None:
            continue
        for geom in body.findall("geom"):
            if geom.get("mesh") and geom.get("class") not in ("foot", "collision"):
                mesh = copy.deepcopy(geom)
                mesh.attrib.pop("class", None)
                mesh.set("type", "mesh")
                mesh.set("group", "2")
                mesh.set("contype", "0")
                mesh.set("conaffinity", "0")
                mesh.set("mass", "0")
                target.append(mesh)

    actuators = ET.SubElement(mjcf, "actuator")
    for side in ("FR", "FL", "RR", "RL"):
        for part in ("hip", "thigh", "calf"):
            name = f"{side}_{part}"
            ET.SubElement(actuators, "motor", {"name": name, "joint": name + "_joint", "ctrlrange": "-23.5 23.5"})
        name = f"{side}_wheel"
        ET.SubElement(actuators, "motor", {"name": name, "joint": name + "_joint", "ctrlrange": "-23.5 23.5"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mjcf).write(output_path, encoding="utf-8", xml_declaration=True)
    check = mujoco.MjModel.from_xml_path(str(output_path))
    if (check.nq, check.nu) != (23, 16):
        raise RuntimeError(f"Unexpected imported model dimensions: nq={check.nq}, nu={check.nu}")
    print(f"Imported {urdf_path} -> {output_path}; mass={check.body_mass.sum():.5f} kg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.urdf, args.output)


if __name__ == "__main__":
    main()
