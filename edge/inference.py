from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.load_model("cap_binaryCNN_first.xmodel")

import numpy as np

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
x = np.load("test_inputs.npy").astype(np.float32)
y = np.load("test_labels.npy")
print(x.shape)  # should be (batch, 1, 83)

# quant
scale = 2 ** fix_point
x_q = (x * scale).round().astype(np.int8)


# set input data
inputData = [x_q]



# %%time
job_id = dpu.execute_async(inputData, outputData)
dpu.wait(job_id)

activations = outputData[0][0]
print(activations)

logit = activations / scale
prob = 1 / (1 + np.exp(-logit))
pred = (prob > 0.5).astype(int)

import time
start = time.time()
job_id = dpu.execute_async(inputData, outputData)
dpu.wait(job_id)
end = time.time()
latency_ms = (end - start) * 1000
print("Latency (ms):", latency_ms)


N = 100   # or 200, small
start = time.time()

for _ in range(N):
    job_id = dpu.execute_async(inputData, outputData)
    dpu.wait(job_id)

end = time.time()
total_ms = (end - start) * 1000

fps = N / (end - start)
print("Throughput (FPS):", fps)
print("Avg latency (ms):", total_ms / N)
