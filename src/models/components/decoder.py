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

        self.dense = nn.Linear(self.latent_dim, self.hidden_unit)
        # cifar10
        self.dense1 = nn.Linear(self.hidden_unit, 8192)
        self.deconv1 = nn.ConvTranspose2d(self.hidden_unit, 32,
                                          kernel_size=self.kernel_size,
                                          stride=2 if self.channels == 1 else 1,
                                          padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 32,
                                          kernel_size=self.kernel_size,
                                          stride=2 if self.channels == 1 else 1,
                                          padding='same')
        self.deconv3 = nn.ConvTranspose2d(32, 32,
                                          kernel_size=self.kernel_size,
                                          stride=2 if self.channels == 1 else 1,
                                          padding='same')
        self.deconv4 = nn.ConvTranspose2d(32, self.channels,
                                          kernel_size=3,
                                          padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=2, padding='same')

    def forward(self, x, y):
        # conditional VAE
        x = torch.cat([x, y], dim = 1)
        x = F.relu(self.dense(x))
        if self.channels == 1:
            x = x.view(-1, self.latent_dim, 1, 1)
        else:
            x = F.relu(self.dense1(x))
            x = x.view(-1, self.latent_dim, 16, 16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        if self.channels == 1:
            x = torch.sigmoid(self.deconv4(x))
        else:
            x = torch.sigmoid(self.conv1(x))
        return x