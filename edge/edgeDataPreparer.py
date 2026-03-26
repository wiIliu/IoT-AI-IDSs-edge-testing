import torch
import numpy as np
### Binary
# from binaryTestLoader import test_loader
# classif_type="bianry"
### Multi
from multiTestLoader import test_loader
classif_type="multi"

# get ONE BATCH from test loader - can change to more
x_batch=[]
y_batch=[]
for x, y in test_loader:
    if len(x) < 64:
        break
    x_batch.append(x.numpy().astype(np.float32))
    y_batch.append(y.numpy().astype(np.int64)) 

print(type(x_batch))
print(x_batch[-1].shape)

# # x_batch: (batch, 1, 83)
# # quantizer expects raw float before quantization
# x_np = x_batch.numpy().astype(np.float32)
# y_np = y_batch.numpy().astype(np.int64)
x_np = np.stack(x_batch, axis=0)
y_np = np.stack(y_batch, axis=0)

print(x_np.shape)

np.save(f"b{classif_type}_test_inputs.npy", x_np)
np.save(f"b{classif_type}_test_labels.npy", y_np)

print(f"Saved {classif_type}_test_inputs.npy and {classif_type}_test_labels.npy")
