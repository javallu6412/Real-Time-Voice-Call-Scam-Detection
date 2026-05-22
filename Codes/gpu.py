import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))


import torch
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))





import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
