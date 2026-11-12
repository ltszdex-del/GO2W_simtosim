# Third-party components and attribution

This project uses standard PPO, GAE, DAgger, symmetry constraints and reference-motion imitation. It does not claim these algorithms as original inventions. Application-specific work includes task design, control integration, simulator adaptation, diagnostics and targeted distillation.

## Unitree robot descriptions and render meshes

Robot descriptions and meshes originate from Unitree Robotics. The render dependency under `.tmp_unitree_mujoco/unitree_robots/go2w/` was obtained from [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), revision `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`. Its BSD 3-Clause license is retained in [third_party/unitree_mujoco/LICENSE](third_party/unitree_mujoco/LICENSE).

Robot assets embedded in the archived training/runtime directories are retained as checkpoint dependencies, including the local collision-geometry migration documented in the simulator. Original source/asset bytes are used for checkpoint hash checks. Ownership of third-party assets remains with their respective holders.

## Runtime and training dependencies

- [MuJoCo](https://github.com/google-deepmind/mujoco): physics, model loading and visualization.
- [PyTorch](https://github.com/pytorch/pytorch): networks, automatic differentiation and optimizers.
- [NumPy](https://github.com/numpy/numpy) and [GLFW Python bindings](https://github.com/FlorianRhiem/pyGLFW): array operations and window/input handling.
- NVIDIA Isaac Sim / Isaac Lab / PhysX: original training environment; installation is external and is not bundled here.
- OpenUSD and SciPy: optional terrain conversion and geometry processing.

These tools retain their own licenses. No blanket license has been applied to all archived code, learned weights and third-party assets in this initial project snapshot.
