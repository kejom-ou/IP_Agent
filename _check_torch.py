import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print(f"Compute Capability: {cap[0]}.{cap[1]}")
print("CUDA Available:", torch.cuda.is_available())
