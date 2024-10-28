import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var, training):
        sigma_epsilon = 1.0 if training else 0.1
        epsilon = torch.randn_like(z_mean) * sigma_epsilon
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
class Encoder(nn.Module):
    def __init__(self, latent_dim=32, image_size:int=28, channels:int=1, kernel_size:int=7, training:bool=True):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.training = training

        if self.channels == 1:
            self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=self.kernel_size, padding='same', dilation=2)
        else:
            self.conv1 =nn.Conv2d(self.channels, 32, kernel_size=self.kernel_size, padding='same')

        self.conv2 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, dilation=2, padding='same')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, dilation=2, padding='same')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, dilation=2, padding='same')

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.latent_dim * self.image_size * self.image_size + 10, 1024)
        self.dense2 = nn.Linear(1024, 1024)

        self.dense3 = nn.Linear(1024, self.latent_dim)
        self.dense4 = nn.Linear(1024, self.latent_dim)
        self.softplus = nn.Softplus()
        self.sampling = Sampling()

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if self.channels == 1:
            x = F.relu(self.conv4(x))
            
        x = self.flatten(x)
        x = torch.cat([x, y], dim=-1)
        
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        z_mean = self.dense3(x)
        z_log_var = self.softplus(self.dense4(x))
        z = self.sampling(z_mean, z_log_var, self.training)
        return z_mean, z_log_var, z
    