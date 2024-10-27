import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifer(nn.Module):
    def __init__(self, channels, image_size):
        super(Classifer, self).__init__()
        self.channels = channels
        self.image_size = image_size

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=0, stride=1)
        # mnist
        if channels == 1:
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1)
        # cifar10   
        elif channels == 3:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=1, dilation=2, stride=1)
          
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 10 * 19, 128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 10)
        # cifar10
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.channels == 3:
            x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        if self.channels == 3:
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            x = self.fc2(x)

        return x