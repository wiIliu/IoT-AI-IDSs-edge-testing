import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


test=pd.read_csv("./data/preprocessedTest.csv")
test['f_Header_b_payload_Ratio'] = test['f_Header_b_payload_Ratio'].replace(np.nan, test['f_Header_b_payload_Ratio'].max() + 1) 
test['b_Header_f_payload_Ratio'] = test['b_Header_f_payload_Ratio'].replace(np.nan, test['b_Header_f_payload_Ratio'].max() + 1) 


# binary class classification
def to_binary_label(y):
    # 0 = benign, 1 = malicious
    benign = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
    return [0 if val in benign else 1 for val in y]

y_test = to_binary_label(test["Attack_type"])

X_test = test.drop("Attack_type", axis=1).values


X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32) 
X_test_tensor = X_test_tensor.unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=64)
class_names = ['benign','malicious']

for x, y in test_loader:
    print(type(x))
    
    print(x.shape)
    print(y.shape)
    break
print('done')
