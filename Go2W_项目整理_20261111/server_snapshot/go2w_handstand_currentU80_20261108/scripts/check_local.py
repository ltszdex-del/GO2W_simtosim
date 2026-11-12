"""语法 + 纯数学检查；不导入 Isaac，不修改现有项目。"""
import ast
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for path in ROOT.rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
print("PASS: Python syntax",flush=True)
subprocess.run([sys.executable,str(ROOT/"tests/test_core.py")],check=True)
