import torch
import torch.nn as nn
import torch.nn.functional as F

#small nn model 3 layers no cnn
class FGNN(nn.Module):
    def __init__(self, input_length=32):
        super(FGNN, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_length, 12)
        self.fc2 = nn.Linear(12, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x