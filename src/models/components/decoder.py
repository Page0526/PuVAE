import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int=32, 
                 channels: int=1, 
                 image_size:int=28, 
                 kernel_size:int=7, 
                 hidden_unit:int=512):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.hidden_unit = hidden_unit

        self.fc = nn.Linear(32 + 10, self.hidden_unit)  # Adjust based on latent_dim
        self.deconv1 = nn.ConvTranspose2d(self.hidden_unit, 32, kernel_size=kernel_size, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, y):
      
        # Concatenate latent vector x and condition vector y
        x1 = torch.cat((x, y), dim=1)  # Assuming y is one-hot encoded with shape [batch_size, 10]
        x2 = F.relu(self.fc(x1))

        # Reshape to [batch_size, 32, 1, 16] for first deconvolution layer
        x3 = x2.view(-1, self.hidden_unit, 1, 1)  # Adjust to match your deconvolution input
        
        # Transpose convolutions
        x4 = F.relu(self.deconv1(x3))  # Output shape: [batch_size, 32, 7, 32]
        x5 = F.relu(self.deconv2(x4))  # Output shape: [batch_size, 32, 14, 14]
        x6 = F.relu(self.deconv3(x5))  # Output shape: [batch_size, 32, 28, 28]
        x7 = self.deconv4(x6)           # Output shape: [batch_size, 1, 28, 28]
        
        return torch.sigmoid(x7)