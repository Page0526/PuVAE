import torch
from torch import nn
import torch.nn.functional as F
from . import Encoder, Decoder

class PuVAE(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        latent_dim: int = 32,
        channels: int=1,
        image_size: int=32
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super(PuVAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()
    
    def forward(self, x, y) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        z_mean, z_log_var, z = self.encoder(x, y)
        reconstructions = self.decoder(z, y) # [128, 1, 28, 28]

        return z_mean, z_log_var, reconstructions


if __name__ == "__main__":
    _ = PuVAE()
