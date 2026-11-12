"""Command-routed policies with synchronized observation and action histories."""
import json
from wasd_adapter import WasdAdapter


class SpecialistAdapter:
    def __init__(self, spec, device):
        self.adapters = {'mixed': WasdAdapter(spec, device)}
        for name, item in spec['specialists'].items():
            self.adapters[name] = WasdAdapter({**spec, **item}, device)
        self.selected = 'straight'
        self.path = spec['checkpoint']
        self.manifest = self.adapters['mixed'].manifest
        for adapter in self.adapters.values():
            assert adapter.manifest['phase']['period'] == self.manifest['phase']['period'] == .30
            for key in ('joint_order', 'output', 'pd', 'limits', 'timing', 'asset_sha256', 'observation_order'):
                assert adapter.manifest[key] == self.manifest[key], key

    def __getattr__(self, name):
        return getattr(self.adapters[self.selected], name)

    def load(self):
        for adapter in self.adapters.values():
            adapter.load()
        print('[SPECIALISTS_LOADED] ' + json.dumps({k: {'checkpoint': a.path, 'iteration': a.iteration}
              for k, a in self.adapters.items()}), flush=True)

    def bind_env(self, env):
        for adapter in self.adapters.values():
            adapter.bind_env(env)

    def reset(self, state):
        self.selected = 'straight'
        for adapter in self.adapters.values():
            adapter.reset(state)

    @staticmethod
    def select(command, previous):
        x, y, yaw = command
        translation = max(abs(x), abs(y)) > .02
        turning = abs(yaw) > .02
        if turning:
            return 'mixed' if translation else 'qe'
        if translation:
            return 'multi' if abs(y) > .02 else 'straight'
        return previous

    def observe(self, state):
        selected = self.select(state['command']['command_m_s_rad_s'], self.selected)
        if selected != self.selected:
            print('[SPECIALIST_SELECTED] ' + selected, flush=True)
        self.selected = selected
        observations = {name: adapter.observe(state) for name, adapter in self.adapters.items()}
        return observations[self.selected]

    def infer(self, observation):
        return self.adapters[self.selected].infer(observation)

    def apply(self, action, dt):
        outputs = {name: adapter.apply(action, dt) for name, adapter in self.adapters.items()}
        return outputs[self.selected]

    def is_complete(self, state):
        return False


def create(spec, device):
    return SpecialistAdapter(spec, device)
