import torch
import numpy as np
### Binary
# from binaryTestLoader import test_loader
# classif_type="bianry"
### Multi
from multiTestLoader import test_loader
classif_type="multi"

# get ONE BATCH from test loader - can change to more
x_batch, y_batch = next(iter(test_loader))

# x_batch: (batch, 1, 83)
# quantizer expects raw float before quantization
x_np = x_batch.numpy().astype(np.float32)
y_np = y_batch.numpy().astype(np.int64)

print(x_np.shape)

np.save(f"{classif_type}_test_inputs.npy", x_np)
np.save(f"{classif_type}_test_labels.npy", y_np)

print(f"Saved {classif_type}_test_inputs.npy and {classif_type}_test_labels.npy")
