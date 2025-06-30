# cuda test
import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is NOT available.")