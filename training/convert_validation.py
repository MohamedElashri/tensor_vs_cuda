import pickle
import numpy as np

# Load validation data
with open("../data/validation_data.pkl", "rb") as f:
    validation_data = pickle.load(f)

# Extract images and labels separately and save as binary files
images = np.concatenate([img.numpy() for img, _ in validation_data])
labels = np.concatenate([lbl.numpy() for _, lbl in validation_data])

images.tofile("../data/validation/validation_images.bin")
labels.tofile("../data/validation/validation_labels.bin")

print("Saved validation images to 'validation_images.bin' and labels to 'validation_labels.bin'")
