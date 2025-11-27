import time
import numpy as np
from pynq_dpu import DpuOverlay

overlay = DpuOverlay("dpu.bit")

### Binary
# overlay.load_model("cap_binaryCNN_first.xmodel")
# obj = "binary"
### Multiclass
overlay.load_model("cap_multiCNN_first.xmodel")
obj = "multi"


dpu = overlay.runner

inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

outputDim = tuple(dpu.get_output_tensors()[0].dims)
outputData = [np.empty(outputDim, dtype=np.int8)]

inputDim = tuple(dpu.get_input_tensors()[0].dims)
inputData = [np.empty(inputDim, dtype=np.int8)]

print(inputTensors)
print(inputTensors[0].dims)
print(inputTensors[0].dtype)

print(outputTensors)
print(outputTensors[0].dims)
print(outputTensors[0].dtype)


fix_point = dpu.get_input_tensors()[0].get_attr("fix_point")
print(fix_point)


# load data
x = np.load(f"{obj}_test_inputs.npy").astype(np.float32)
y = np.load(f"{obj}_test_labels.npy")
print(x.shape)  # should be (batch, 1, 83)


# quantize input data
scale = 2 ** fix_point
x_q = (x * scale).round().astype(np.int8)


# set input data
batchNumber = 2
inputData = [x_q[batchNumber]] # can be any of the batches 0-63 from x_q
print(y[batchNumber]) # correct label


####### LATENCY #######
total = 0
inputData_b = [x_q[batchNumber]]
for s in range(2000):
    start = time.perf_counter()
    job = dpu.execute_async(inputData_b, outputData)
    dpu.wait(job)
    end = time.perf_counter()
    total+=(end-start)
#     print(f"Sample {b}: {(end-start)*1000:.6f} ms")
print(f"Per sample average: {(total/2000)*1000:.6f} ms")


# single instance latency timing. Output: wall time | cpu time
# % % time
job_id = dpu.execute_async(inputData_b, outputData)
dpu.wait(job_id)


# check predictions
activations = outputData[0][0]
print(activations)
if obj == 'binary':
    logit = activations / scale
    prob = 1 / (1 + np.exp(-logit))
    pred_class = (prob > 0.5).astype(int)
else:
    # probabilities - softmax
    logits = activations / scale   # only if your model is quantized
    exp = np.exp(logits - np.max(logits))  # stability
    probs = exp / np.sum(exp)

    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    print("Confidence:", confidence, "\n", probs)

print("Pred class:", pred_class)


####### THROUGHPUT #######
for b in [1, 2, 8, 16, 32, 64]:
    batch = x_q[:b].astype(np.float32)
    inputData = [batch]

    out_shape = dpu.get_output_tensors()[0].dims
    out_shape[0] = b
    print(out_shape)
    outputData = [np.empty(out_shape, dtype=np.float32)]
    start = time.perf_counter()
    job = dpu.execute_async(inputData, outputData)
    dpu.wait(job)
    end = time.perf_counter()

    latency = end - start
    print(f"Batch {b}: {latency*1000:.3f} ms | Per sample: {(latency/b)*1000:.3f} ms")



