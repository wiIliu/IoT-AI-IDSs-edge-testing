import torch.nn as nn
import torch.nn.functional as F

class MultiClass1dCNN(nn.Module):
    def __init__(self, num_features=83, num_classes=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,3), padding=(0,1))
        
        self.pool = nn.AvgPool2d((1,2))
        self.fc1 = nn.Linear(32*(num_features//2),128)
        # self.dropout = nn.Dropout1d(.3)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
        # convolutions
        x = x.view(x.size(0), 1, 1, 83)

        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.pool(x)
        # dense layers
        x = x.reshape(x.size(0),-1)
        x = self.fc2(F.relu(self.fc1(x)))
        # x = self.fc2(self.dropout(F.relu(self.fc1(x))))

        return x
    
    print("done")
    