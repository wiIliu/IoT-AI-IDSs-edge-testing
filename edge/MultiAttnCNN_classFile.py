import torch.nn as nn
import torch.nn.functional as F

class MultiAttnCNN(nn.Module):
    def __init__(self, num_classes=12, num_features=83):
        super(MultiAttnCNN, self).__init__()
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.attn_embed_dim = 32
        self.num_heads = 8
        self.attn = nn.MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=self.num_heads, batch_first=True)
        self.fc1 = nn.Linear(32*(num_features//2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        B, C, L = x.shape  # Batch, Channels, Length
        x = x.permute(0, 2, 1)  # (B, L, C) for attention
        attn_out, _ = self.attn(x, x, x)
        x = attn_out + x  # Residual connection
        x = x.permute(0, 2, 1).contiguous().view(B, -1)  # Back to (B, C*L)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


