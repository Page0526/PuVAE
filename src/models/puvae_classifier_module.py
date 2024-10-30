from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.puvae_classifier import PuVAEClassifier

class PuVAEModule(LightningModule):
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
        net: PuVAEClassifier,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        threshold:float=0.12,
        ce_coeff:float=1.0,
        rc_coeff:float=1.0,
        kl_coeff:float=1.0
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.ce_coeff = ce_coeff
        self.rc_coeff = rc_coeff
        self.kl_coeff = kl_coeff

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # ensure the classifier is not trained
        for param in self.net.classifier.parameters():
            param.requires_grad = False

    def forward(self, x, y) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, y)

    def best_reconstruction(self, x, y, n_classes: int=10):
        '''
        Identify best reconstruction from puvae
        '''
        batch_size = x.shape[0]
        
        images = x.repeat_interleave(n_classes, dim=0) # repeat elements of a tensor
        labels = torch.eye(n_classes, device=x.device).repeat(batch_size, 1)

        # from IPython import embed
        # embed()

        _, _, reconstructions = self.net.puvae(images, labels)
        errors = F.mse_loss(reconstructions, images, reduction='none').mean(dim=[2, 3])
        errors = errors.view(batch_size, n_classes)

        best_idxs = errors.argmin(dim=1) + torch.arange(0, batch_size, dtype=torch.int64, device=errors.device) * n_classes 
        best_reconstructions = reconstructions[best_idxs]

        return best_reconstructions, errors.min(dim=1).values

    def predict(self, x, y):
        '''
        Identify clean and adv predictions
        '''
        best_reconstruction, errors = self.best_reconstruction(x, y)
        preds = self.net.classifier(best_reconstruction)
        
        keep_preds = (errors < self.threshold).float()
        new_preds = preds * keep_preds.unsqueeze(-1)
        adv_column = (errors >= self.threshold).float().unsqueeze(-1)

        new_preds = torch.cat([new_preds, adv_column], dim=1)
        return new_preds

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

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
        z_mean, z_log_var, reconstruction, preds = self.forward(x, y)
        kl_loss = torch.mean(z_mean**2 + torch.exp(z_log_var) - 1 - z_log_var)

        # make sure x and reconstruction in range [0, 1]
        x = torch.clamp(x.float(), min=0, max=1)
        reconstruction = torch.clamp(reconstruction.float(), min=0, max=1)

       # Move tensors to CPU for debug-friendly errors if running on CUDA
        if x.is_cuda:
            x_cpu = x.cpu()
            reconstruction_cpu = reconstruction.cpu()
        else:
            x_cpu = x
            reconstruction_cpu = reconstruction

        try:
            # Attempt to calculate rc_loss on CPU
            rc_loss = F.binary_cross_entropy(reconstruction_cpu, x_cpu, reduction='mean')
        except RuntimeError as e:
            print("RuntimeError in binary_cross_entropy for rc_loss on CPU")
            raise  # Re-raise the RuntimeError
        except ValueError as e:
            print("ValueError in binary_cross_entropy for rc_loss on CPU")
            raise

        # Continue with other loss calculations (these can stay on CUDA)
        ce_loss = F.cross_entropy(preds.float(), y.float())
        loss = self.ce_coeff * ce_loss + self.rc_coeff * rc_loss + self.kl_coeff * kl_loss

        from IPython import embed
        embed()

        
        return preds, loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        preds, loss = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
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
        
        loss, preds = self.model_step(batch)

        best_reconstructions = self.best_reconstruction(x, y)
        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

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
        loss, preds, = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
    _ = PuVAEModule(None, None, None, None)
