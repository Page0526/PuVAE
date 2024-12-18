from typing import Any, Dict, Tuple
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification.accuracy import Accuracy
from torchvision.utils import make_grid
from src.models.components.classifier import Classifier
from src.models.components.cvae import ConditionalVAE
import torchattacks

class PuVAEModule(LightningModule):

    def __init__(
        self,
        net: ConditionalVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        classifier: Classifier,
        classifier_ckpt: str = "/mnt/apple/k66/ptrang/PuVAE/logs/train/runs/2024-12-18_13-03-55/checkpoints/last.ckpt",
        threshold:float=0.12,
        ce_coeff:float=10,
        rc_coeff:float=1,
        kl_coeff:float=0.0025
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.threshold = threshold
        self.classifier = classifier
        self.load_classifier(classifier_ckpt)
        
        self.ce_weight = ce_coeff
        self.rc_weight = rc_coeff
        self.kl_weight = kl_coeff

        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.train_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x, y) -> torch.Tensor:
        recons, input, mu, log_var = self.net(x, y)
        loss, rc_loss, kl_loss = self.net.loss_function(recons, input, mu, log_var)

        return recons, loss, rc_loss, kl_loss
    
    def load_classifier(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        checkpoint_state_dict = checkpoint['state_dict']
        checkpoint_state_dict = {k.replace('net.', ''): v for k, v in checkpoint_state_dict.items()}
        
        self.classifier.load_state_dict(checkpoint_state_dict)
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device)

        for param in self.classifier.parameters():
            param.requires_grad = False

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x, y = batch
        reconstruction, kl_loss = self.net(x, y)

        # calculate loss
        rc_loss = self.bce(reconstruction, x)
        vae_loss = self.rc_coeff * rc_loss + self.kl_coeff * kl_loss

        return reconstruction, vae_loss

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:   
        x, y = batch
        recons, loss, recons_loss, kl_loss = self.forward(x, y)

        preds = self.classifier(recons)
        ce_loss = F.cross_entropy(preds, y.float())
        
        total_loss = loss + self.ce_weight*ce_loss
        self.train_loss(total_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx%10 == 0:
            psnr_value = self.train_psnr(recons, x)
            ssim_value = self.train_ssim(recons, x)
            
            reconstruction = make_grid(recons, nrow=10, normalize=True)
            x_grid = make_grid(x, nrow=10, normalize=True)
            preds_labels = preds.argmax(dim=1)
            y_labels = y.argmax(dim=1)

            captions = [
            "Reconstruction/" + ', '.join(map(str, preds_labels.tolist())),
            'real/' + ', '.join(map(str, y_labels.tolist())),
            ]
            self.logger.log_image(key='train/image', images=[reconstruction, x_grid], caption=captions)   
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.train_psnr.reset()
        self.train_ssim.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:    
        x, y = batch
        recons, loss, recons_loss, kl_loss = self.forward(x, y)
        
        preds = self.classifier(recons)
        ce_loss = F.cross_entropy(preds, y.float())

        total_loss = loss + self.ce_weight*ce_loss
        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx%10 == 0:
            psnr_value = self.val_psnr(recons, x)
            ssim_value = self.val_ssim(recons, x)
            
            reconstruction = make_grid(recons, nrow=10, normalize=True)
            x_grid = make_grid(x, nrow=10, normalize=True)
            
            preds_labels = preds.argmax(dim=1)
            y_labels = y.argmax(dim=1)
            
            captions = [
            "Reconstruction/" + ', '.join(map(str, preds_labels.tolist())),
            'real/' + ', '.join(map(str, y_labels.tolist())),
            ]
            self.logger.log_image(key='val/image', images=[reconstruction, x_grid], caption=captions) 
            self.log("val/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_psnr.reset()
        self.val_ssim.reset()


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        recons, loss, recons_loss, kl_loss = self.forward(x, y)

        # self.classifier = self.load_classifier("/kaggle/input/classifier/pytorch/fashion_classifier/1/fashion_classifier.ckpt")
        preds = self.classifier(recons)
        ce_loss = F.cross_entropy(preds, y.float())
        
        total_loss = loss + self.ce_weight*ce_loss
        self.test_loss(total_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx%10 == 0:
            
            reconstruction = make_grid(recons, nrow=10, normalize=True)
            x_grid = make_grid(x, nrow=10, normalize=True)
            
            preds_labels = preds.argmax(dim=1)
            y_labels = y.argmax(dim=1)
            captions = [
            "Reconstruction/" + ', '.join(map(str, preds_labels.tolist())),
            'real/' + ', '.join(map(str, y_labels.tolist())),
            ]
            self.logger.log_image(key='test/image', images=[reconstruction, x_grid], caption=captions) 
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PuVAEModule(None, None, None, None)
