import subprocess, sys

commands = [
    [sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'torch'],
    [sys.executable, '-m', 'pip', 'install', 'torch==2.0.1', 'transformers==4.31.0', 'datasets==2.14.0', 'wandb==0.21.0'],
]

for cmd in commands:
    subprocess.check_call(cmd)

import torch
assert torch.cuda.is_available(), "GPU not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")

