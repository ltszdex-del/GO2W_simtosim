from pathlib import Path
import importlib.util
import json
import sys
import torch

kind = sys.argv[1]
base = Path('/home/ubuntu/Go2W')
live = base / 'interactive_latest_test_20261110'
control = base / 'refine_followup_20261111'
root = base / f'go2w_{kind}_followup_20261111'
status = json.loads((control / f'{kind}_status.json').read_text())
if status['state'] != 'complete':
    status['checkpoint']=(Path(status['log'])/'latest.txt').read_text().strip()
    assert int(Path(status['checkpoint']).stem.split('_')[-1])==status['goal_iteration']
sys.path.insert(0, str(live))
sys.path.insert(0, str(root))
from learning import Policy
torch.set_num_threads(4)
ck = torch.load(status['checkpoint'], map_location='cpu', weights_only=False)
model = Policy().eval()
model.load_state_dict(ck['model'])
# Training and runtime each have a commands.py with different interfaces.
sys.path.remove(str(root))
sys.modules.pop('commands', None)
cfg = json.loads((live / 'config.json').read_text())
mode = 'wheel_locomotion' if kind == 'wheel' else 'locomotion'
spec = dict(cfg['policies'][mode], checkpoint=status['checkpoint'], training_root=str(root))
name = 'wheel_mirror_brake_adapter' if kind == 'wheel' else 'step_followup_adapter'
path = live / (name + '.py') if kind=='wheel' else control / (name+'.py')
loader = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
adapter = module.create(spec, 'cpu')
adapter.load()
torch.manual_seed(1042)
obs = torch.randn(128, 570) * .3
with torch.no_grad():
    error = float((model(obs) - adapter.infer(obs)).abs().max())
assert error < 1e-6, error
result = dict(kind=kind, checkpoint=status['checkpoint'], action_parity_error=error)
(control / f'{kind}_final_parity.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
