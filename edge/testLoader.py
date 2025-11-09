import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


test=pd.read_csv("./data/preprocessedTest.csv")
test['f_Header_b_payload_Ratio'] = test['f_Header_b_payload_Ratio'].replace(np.nan, test['f_Header_b_payload_Ratio'].max() + 1) 
test['b_Header_f_payload_Ratio'] = test['b_Header_f_payload_Ratio'].replace(np.nan, test['b_Header_f_payload_Ratio'].max() + 1) 


# Multi class classification
label_encoder = LabelEncoder()
label_encoder.fit(test["Attack_type"])

y_test = label_encoder.transform(test["Attack_type"])

# Get class names for later

X_test = test.drop("Attack_type", axis=1).values

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long) 

def pad_to_square(X, size=90):
    pad_len = size - X.shape[1]
    if pad_len > 0:
        X = F.pad(X, (0, pad_len))
        return X


X_test_padded = pad_to_square(X_test_tensor)

X_test_cnn = X_test_padded.view(-1, 1, 9, 10)


test_dataset = TensorDataset(X_test_cnn, y_test_tensor)


class_names = label_encoder.classes_
test_loader = DataLoader(test_dataset, batch_size=64)

for x, y in test_loader:
    print(type(x))
    
    print(x.shape)
    print(y.shape)
    break
print('done')
