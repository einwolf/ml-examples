import sys

import torch

print(f"Python version {sys.version}")
print(f"Pytorch version {torch.__version__}")

print(f"{torch.cuda.is_available() = }")
print(f"{torch.backends.cudnn.version() = }")
print(f"{torch.cuda.device_count() = }")
print(f"{torch.cuda.current_device() = }")

for i in range(torch.cuda.device_count()):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
