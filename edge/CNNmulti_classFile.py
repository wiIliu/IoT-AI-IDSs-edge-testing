import torch.nn as nn
import torch.nn.functional as F

class MultiClassAttackCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(MultiClassAttackCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.flatten_dim = 32 * 4 * 5  # after pooling
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)  # Output layer for 12 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
