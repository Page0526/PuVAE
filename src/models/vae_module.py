from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification.accuracy import Accuracy
from torchvision.utils import make_grid
from src.models.components.puvae import PuVAE
import torch.nn.functional as F


class VAEModule(LightningModule):

    def __init__(
        self,
        net: PuVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        rc_coeff:float=0.01,
        kl_coeff:float=0.1
    ) -> None:

        super().__init__()

        
        self.save_hyperparameters(logger=False)

        self.net = net

        
        self.criterion = torch.nn.BCELoss()
        self.rc_coeff = rc_coeff
        self.kl_coeff = kl_coeff

        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        
        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x: torch.Tensor, y) -> torch.Tensor:
        
        return self.net(x, y)

    def on_train_start(self) -> None:
        
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x, y = batch
        reconstructions, kl_loss = self.forward(x, y)

        rc_loss = self.criterion(reconstructions, x)
        loss = rc_loss * self.rc_coeff + kl_loss * self.kl_coeff

        return reconstructions, loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        

        reconstructions, loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y = batch
        reconstruction, loss = self.model_step(batch)
        # Compute PSNR and SSIM
        psnr_value = self.val_psnr(reconstruction, x)
        ssim_value = self.val_ssim(reconstruction, x)
            
        # update and log metrics
        self.val_loss(loss)

        if batch_idx%10 == 0:
            reconstruction = make_grid(reconstruction, nrow=10, normalize=True)
            x = make_grid(x, nrow=10, normalize=True)
            self.logger.log_image(key='val/image', images=[reconstruction, x], caption=['reconstruction','real'])        

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y = batch
        reconstruction, loss = self.model_step(batch)

        # Compute PSNR and SSIM
        psnr_value = self.test_psnr(reconstruction, x)
        ssim_value = self.test_ssim(reconstruction, x)

        # update and log metrics
        self.test_loss(loss)

        if batch_idx%10 == 0:
            reconstruction = make_grid(reconstruction, nrow=10, normalize=True)
            x = make_grid(x, nrow=10, normalize=True)
            self.logger.log_image(key='test/image', images=[reconstruction, x], caption=['reconstruction','real'])        

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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
    _ = VAEModule(None, None, None)