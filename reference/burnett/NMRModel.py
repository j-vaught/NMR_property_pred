import torch
import torch.nn as nn
import torch.nn.functional as F


#load A and B model
class NMR1DCNN(nn.Module):
    def __init__(self, input_length=6554):
        super(NMR1DCNN, self).__init__()
        
        # Conv1D layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        # MaxPooling layer 1
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        # MaxPooling layer 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        # MaxPooling layer 3
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 4
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        # MaxPooling layer 4
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the flattened size after the last convolution and pooling layers
        self.flattened_size = self.calculate_flattened_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def calculate_flattened_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x