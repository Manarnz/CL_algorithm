import torch
import torch.nn as nn
import torch.nn.functional as F
import setting

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv3d(10, 80, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2,2)
        self.fc1 = nn.Linear(32 * 128 * 128, 64) 
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 128 * 128)  # Flatten the output for fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    