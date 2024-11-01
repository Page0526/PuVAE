from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification.accuracy import Accuracy
from torchvision.utils import make_grid
from src.models.components.puvae_classifier import PuVAE
from src.models.components.classifier import Classifier

class PuVAEClassifierModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: PuVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        classifier: Classifier,
        pretrained_classifier_path: str = "/mnt/apple/k66/ptrang/PuVAE/logs/train/runs/2024-11-01_21-46-18/checkpoints/last.ckpt",  # path to your pre-trained classifier
        threshold:float=0.12,
        ce_coeff:float=10,
        rc_coeff:float=0.01,
        kl_coeff:float=0.1
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.threshold = threshold
        # Load pre-trained classifier weights
        self.load_pretrained_classifier(pretrained_classifier_path)
        self.classifier = classifier
        self.criterion = torch.nn.CrossEntropyLoss() # for classifier
        self.bce = torch.nn.BCELoss() # for reconstruction
        self.ce_coeff = ce_coeff
        self.rc_coeff = rc_coeff
        self.kl_coeff = kl_coeff

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()

    def load_pretrained_classifier(self, path: str):
        """Load pre-trained weights for the classifier."""
        try:
            pretrained_state_dict = torch.load(path, map_location=self.device)
            self.classifier.load_state_dict(pretrained_state_dict)
            print("Pre-trained classifier weights loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-trained classifier weights: {e}")

    def forward(self, x, y) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, y)

    def best_reconstruction(self, batch: Tuple[torch.Tensor, torch.Tensor], n_classes: int=10):
        '''
        Identify best reconstruction from puvae
        '''
        x, y = batch
        batch_size = x.shape[0]
        
        images = x.repeat_interleave(n_classes, dim=0) # repeat elements of a tensor
        labels = torch.eye(n_classes, device=x.device).repeat(batch_size, 1)

        reconstructions, kl_loss = self.net(images, labels)

        errors = F.mse_loss(reconstructions, images, reduction='none').mean(dim=[2, 3])
        errors = errors.view(batch_size, n_classes)

        best_idxs = errors.argmin(dim=1) + torch.arange(0, batch_size, dtype=torch.int64, device=errors.device) * n_classes 
        best_reconstructions = reconstructions[best_idxs]

        return best_reconstructions, errors.min(dim=1).values

    def predict(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        '''
        Identify clean and adv predictions
        '''
        x, y = batch
        best_reconstruction, errors = self.best_reconstruction(batch)
        preds = self.classifier(best_reconstruction)
        
        keep_preds = (errors < self.threshold).float()
        new_preds = preds * keep_preds.unsqueeze(-1)
        adv_column = (errors >= self.threshold).float().unsqueeze(-1)

        new_preds = torch.cat([new_preds, adv_column], dim=1)
        return new_preds

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        reconstruction, kl_loss = self.forward(x, y)

        # calculate loss
        rc_loss = self.bce(reconstruction, x)
        # loss = self.ce_coeff * ce_loss + self.rc_coeff * rc_loss + self.kl_coeff * kl_loss
        vae_loss = self.rc_coeff * rc_loss + self.kl_coeff * kl_loss

        return reconstruction, vae_loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y = batch
        reconstruction, vae_loss = self.model_step(batch)
        preds = self.classifier(reconstruction)

        ce_loss = F.cross_entropy(preds.float(), y.float())
        loss = self.ce_coeff * ce_loss + vae_loss
        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return vae_loss

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
        
        reconstruction, vae_loss = self.model_step(batch)
        # preds = self.classifier(reconstruction)

        # ce_loss = F.cross_entropy(preds.float(), y.float())
        # loss = self.ce_coeff * ce_loss + vae_loss

        # Compute PSNR and SSIM
        psnr_value = self.val_psnr(reconstruction, x)
        ssim_value = self.val_ssim(reconstruction, x)

        # update and log metrics
        self.val_loss(vae_loss)
        # self.val_acc(preds, y)

        if batch_idx%10 == 0:
            reconstruction = make_grid(reconstruction, nrow=10, normalize=True)
            x = make_grid(x, nrow=10, normalize=True)
            self.logger.log_image(key='val/image', images=[reconstruction, x], caption=['reconstruction','real'])        
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y = batch
        reconstruction, vae_loss= self.model_step(batch)
        preds = self.classifier(reconstruction)
        best_reconstruction = self.predict(batch)

        ce_loss = F.cross_entropy(preds.float(), y.float())
        loss = self.ce_coeff * ce_loss + vae_loss
        psnr_value = self.test_psnr(reconstruction, x)
        ssim_value = self.test_ssim(reconstruction, x)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, y)

        # if batch_idx == 0:
        #     # Only log the first few images
        #     num_images = min(8, best_reconstruction.size(0))  # Log up to 8 images
        #     images = best_reconstruction[:num_images]
        #     predicted_labels = preds[:num_images]
        #     true_labels = y[:num_images]
        #     captions = [f"Pred: {pred} | True: {true}" for pred, true in zip(predicted_labels, true_labels)]
        #     # Convert to format suitable for logging, e.g., WandB or TensorBoard
        #     # image_grid = make_grid(images, nrow=4)
        #     self.logger.log_image(key="val/images_with_preds", images=[img for img in images], caption=captions)

        # Log images every 10 batches
        if batch_idx == 0:
            # Make grid for logging images
            reconstruction_grid = make_grid(best_reconstruction, nrow=10, normalize=True)
            x_grid = make_grid(x, nrow=10, normalize=True)
            labels = torch.argmax(y, dim=1)
            
            # Convert labels to strings for captions
            real_labels = [str(label.item()) for label in labels]
            recon_labels = [str(label.item()) for label in torch.argmax(preds, dim=1)]
            
            # Log images with captions
            self.logger.log_image(
                key='test/image',
                images=[reconstruction_grid, x_grid],
                caption=[f'Reconstruction (Labels: {", ".join(recon_labels)})', 
                        f'Real (Labels: {", ".join(real_labels)})']
            )     

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            
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
    _ = PuVAEClassifierModule(None, None, None, None)
