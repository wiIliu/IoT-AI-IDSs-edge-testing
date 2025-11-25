import torch.nn as nn
import torch.nn.functional as F

class BinaryCNN(nn.Module):
    def __init__(self, input_length=83, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout1d(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_length // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.fc1(self.flatten(x)))
        x = self.fc2(self.dropout(x))

        return x
    
print("Done")
