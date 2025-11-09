import torch

from pytorch_nndct.apis import torch_quantizer
from testLoader import test_loader, class_names
from edge.CNNmulti_classFile import MultiClassAttackCNN



def load_model(model_path, device):
    model = MultiClassAttackCNN(num_classes=12)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

cnn=load_model(r"2dcnn_multiclass.pth",'cpu')

print("uuuuu")
input("pause...")


cnn.eval()  # Set the model to evaluation mode

xtestpts, ytestpts = next(iter(test_loader))
print(xtestpts.shape)
xtestpts_0 = xtestpts[0]
xtestpts_0 =xtestpts_0.unsqueeze(1)
with torch.no_grad():
    output = cnn(xtestpts_0)

lbls = class_names
print(type(lbls), lbls)

# Get the top 5 predicted classes and their confidence scores
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# print top 5
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')

print("ytestpts\n")
print(ytestpts)

# pause here
input("Press Enter to continue...")


# prepare images for calibration
print("Prepare for calibration...")
calib_pts = []

print(len(xtestpts))


# go through each subfolder 0 thru 999


for i in range(10):
   calib_pts.append(xtestpts[i+1])

print("Converting to batch...")
calib_batch = torch.stack(calib_pts[0:9])
print("calib batch - ",calib_batch.shape)


print("Quantizing...")
quantizer = torch_quantizer("calib", cnn, (calib_batch))
quant_model = quantizer.quant_model


print("Evaluating quantized model...")


device = torch.device("cpu")
quant_model.eval()
quant_model = quant_model.to(device)
output = quant_model(xtestpts_0)


# Get the top 5 predicted classes and their confidence scores
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)


# print top 5
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')

print("Exporting...")


quantizer.export_quant_config()


print("Deploying...")


# create batch with 1 test image
test = []
test.append(xtestpts[0])
test_batch = torch.stack(test)


# create quantizer with "test" (i.e. evaluation and export) setting
quantizer = torch_quantizer("test", cnn, (test_batch))


# need to eval again
# see https://github.com/Xilinx/Vitis-AI/issues/974#issuecomment-1232952437
quant_model = quantizer.quant_model
device = torch.device("cpu")
quant_model.eval()
quant_model = quant_model.to(device)
output = quant_model(xtestpts_0)


# Get the top 5 predicted classes and their confidence scores
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)


# print top 5
for i in range(top5_prob.size(0)):
   print(f'{i} {top5_prob[i].item() * 100:.2f}%')


quantizer.export_xmodel(deploy_check=True)
quantizer.export_onnx_model()
print("succcesss!!!!")
