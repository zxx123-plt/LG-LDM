import torch


print(f"CUDA available: {torch.cuda.is_available()}")


current_device = torch.cuda.current_device()
print(f"Current CUDA device index: {current_device}")


device_count = torch.cuda.device_count()
print(f"Number of available CUDA devices: {device_count}")

for i in range(device_count):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"  Capability: {torch.cuda.get_device_capability(i)}")