import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, Callback
from torchmetrics import Accuracy
import numpy as np
from src.models.components.puvae_model import PuVAE
from src.models.components.classifier import Classifer

class PuVAEClassifier(LightningModule):
    def __init__(self,
                 puvae: PuVAE,
                 classifier: Classifer):
        super(PuVAEClassifier, self).__init__()
        self.puvae = puvae
        # the classifier need to be trained and frozen weights before training PuVAEClassifier
        self.classifier = classifier
        # weight for each loss in overall loss


    def forward(self, x, y):
        z_mean, z_log_var, reconstruction = self.puvae(x, y)
        preds = self.classifier(reconstruction)
        
        return z_mean, z_log_var, reconstruction, preds



