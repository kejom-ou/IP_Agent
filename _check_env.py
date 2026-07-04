import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
import subprocess
r = subprocess.run(["pip", "show", "torch"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "Version" in line:
        print("pip Version:", line.strip())
