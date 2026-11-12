"""Use the installed global Torch and local MuJoCo, without modifying either."""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SNAPSHOT = WORKSPACE / 'Go2W_项目整理_20261111/server_snapshot'
RUNTIME = ROOT / 'runtime' if (ROOT / 'runtime/config.json').exists() else SNAPSHOT / 'interactive_latest_test_20261110'

def relocate_text(text):
    """Resolve the exported workstation marker and original training prefix."""
    return (text.replace('__GO2W_WORKSPACE__/', WORKSPACE.as_posix() + '/')
                .replace('/home/ubuntu/Go2W/', SNAPSHOT.as_posix() + '/'))
if importlib.util.find_spec('mujoco') is None:
    sys.path.append(str(WORKSPACE / '.venv-mujoco/Lib/site-packages'))
sys.path.insert(0, str(RUNTIME))


class LocalSource(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode('utf-8')
        source = source.replace('/home/ubuntu/Go2W/', SNAPSHOT.as_posix() + '/')
        source = source.replace('.read_text()', '.read_text().replace("/home/ubuntu/Go2W/", ' + repr(SNAPSHOT.as_posix() + '/') + ')')
        return compile(source, self.path, 'exec')


class SnapshotPaths(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith('.py') and (Path(spec.origin).is_relative_to(SNAPSHOT) or Path(spec.origin).is_relative_to(ROOT/'runtime')):
            spec.loader = LocalSource(fullname, spec.origin)
            return spec


sys.meta_path.insert(0, SnapshotPaths())


def install_tensor_compatibility():
    """Only config declarations/math imports. No Isaac physics is emulated here.

    Original embedded controllers run against real MuJoCo state and actuators.
    Config placeholders are never used to create a simulation or training env.
    """
    import torch
    class Config:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def replace(self, **kwargs):
            import copy
            result = copy.deepcopy(self); result.__dict__.update(kwargs); return result
    Config.InitialStateCfg = Config
    def module(name, **values):
        m = types.ModuleType(name); m.__path__ = []; m.__dict__.update(values)
        sys.modules[name] = m
        if '.' in name: setattr(sys.modules[name.rsplit('.', 1)[0]], name.rsplit('.', 1)[1], m)
        return m
    module('isaaclab')
    sim = module('isaaclab.sim')
    for name in ('GroundPlaneCfg', 'RigidBodyMaterialCfg', 'DomeLightCfg', 'UrdfFileCfg',
                 'RigidBodyPropertiesCfg', 'ArticulationRootPropertiesCfg', 'SimulationCfg', 'UsdFileCfg'):
        setattr(sim, name, Config)
    module('isaaclab.assets', ArticulationCfg=Config, AssetBaseCfg=Config)
    module('isaaclab.actuators', ImplicitActuatorCfg=Config)
    class ForbiddenEnv:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('Isaac scene construction is unavailable; use embedded controller only')
    module('isaaclab.envs', DirectRLEnv=ForbiddenEnv, DirectRLEnvCfg=Config)
    module('isaaclab.scene', InteractiveSceneCfg=Config)
    module('isaaclab.sensors', ContactSensorCfg=Config)
    module('isaaclab.utils', configclass=lambda cls: cls)
    def quat_mul(a, b):
        av, bv = a[..., :3], b[..., :3]
        return torch.cat((a[..., 3:] * bv + b[..., 3:] * av + torch.linalg.cross(av, bv),
                          a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)), -1)
    def quat_apply(q, v):
        qv, v = torch.broadcast_tensors(q[..., :3], v)
        uv = torch.linalg.cross(qv, v)
        return v + 2 * (q[..., 3:] * uv + torch.linalg.cross(qv, uv))
    def quat_apply_inverse(q, v):
        return quat_apply(torch.cat((-q[..., :3], q[..., 3:]), -1), v)
    def quat_from_euler_xyz(r, p, y):
        cr,sr,cp,sp,cy,sy = (r/2).cos(),(r/2).sin(),(p/2).cos(),(p/2).sin(),(y/2).cos(),(y/2).sin()
        return torch.stack((sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy), -1)
    module('isaaclab.utils.math', quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse,
           quat_mul=quat_mul, quat_from_euler_xyz=quat_from_euler_xyz)
    # Original env modules import Warp only for training-time randomization.
    # Deliberately expose no Warp functions: accidental training paths fail loudly.
    module('warp')


def load_config():
    import json
    config = json.loads(relocate_text((RUNTIME / 'config.json').read_text(encoding='utf-8')))
    config['bridge_port'] = 8776
    return config
