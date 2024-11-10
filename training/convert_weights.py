import torch
import numpy as np
import os

precision = "fp16"  # or "fp32"
# Load the trained model weights
state_dict = torch.load(f"../data/simple_cnn_{precision}.pth")

# Create the directory if it doesn't exist
os.makedirs("../data/weights", exist_ok=True)

for name, param in state_dict.items():
    # Convert to FP16
    param_fp16 = param.half()  # Convert to FP16
    param_numpy = param_fp16.cpu().numpy()
    
    print(f"Saving {name}:")
    print(f"Shape: {param_numpy.shape}")
    print(f"Dtype: {param_numpy.dtype}")
    print(f"Size in bytes: {param_numpy.nbytes}")
    
    # Save with correct dtype specification
    param_numpy.astype(np.float16).tofile(f"../data/weights/{name}_{precision}.bin")
    
    # Verify the file size
    file_size = os.path.getsize(f"../data/weights/{name}_{precision}.bin")
    print(f"Written file size: {file_size} bytes")
    print()