"""Explicit factory loading; a bare checkpoint path is never a complete adapter."""
import importlib
from pathlib import Path

def load_adapter(spec,root,device):
    if not isinstance(spec,dict) or 'factory' not in spec or 'checkpoint' not in spec:
        raise ValueError('Policy requires {factory: module:factory, checkpoint: path}, not a bare weight path')
    checkpoint=Path(spec['checkpoint']).expanduser()
    if not checkpoint.is_absolute():checkpoint=Path(root)/checkpoint
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    module,name=spec['factory'].split(':',1)
    factory=getattr(importlib.import_module(module),name)
    return factory({**spec,'checkpoint':str(checkpoint.resolve())},device)
