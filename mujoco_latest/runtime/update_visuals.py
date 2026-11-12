"""Compatibility entry: deploy the complete current environment together."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name('deploy_task2.py')),run_name='__main__')
