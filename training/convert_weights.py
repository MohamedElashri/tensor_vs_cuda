import torch
import numpy as np
import os

# Load the trained model weights
state_dict = torch.load("../data/simple_cnn.pth")

# Save each layer's weights as a binary file
# Create the directory if it doesn't exist
os.makedirs("../data/weights", exist_ok=True)

for name, param in state_dict.items():
    param_numpy = param.cpu().numpy()
    param_numpy.tofile(f"../data/weights/{name}.bin")
    print(f"Saved {name} to ../data/weights/{name}.bin")
