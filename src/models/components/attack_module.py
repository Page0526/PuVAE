from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
import foolbox as fb
from foolbox import PyTorchModel

class AttackModule(LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 attack_name:str,
                 epsilon:float)->None:
        super().__init__()
        self.model = model # puvae_classifier
        self.attack_name = attack_name
        self.epsilon = epsilon

        self.fmodel = PyTorchModel(model, bounds=(0, 1))

        self.attacker = self.select_attack()

    def select_attack(self) -> Optional[fb.Attack]:
        """Select the appropriate attack based on the attack name."""
        if self.attack_name == "LinfProjectedGradientDescentAttack":
            return fb.attacks.LinfProjectedGradientDescentAttack()
        elif self.attack_name == "LinfFastGradientAttack":
            return fb.attacks.LinfFastGradientAttack()
        elif self.attack_name == "LinfBasicIterativeAttack":
            return fb.attacks.LinfBasicIterativeAttack()
        elif self.attack_name == "L2BasicIterativeAttack":
            return fb.attacks.L2BasicIterativeAttack()
        elif self.attack_name == "L2ProjectedGradientDescentAttack":
            return fb.attacks.L2ProjectedGradientDescentAttack()
        else:
            raise ValueError(f"Unknown attack name: {self.attack_name}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    
    def model_step(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Return: A tuple of (adv_imgs, ori_imgs)
        '''
        adv_imgs = self.attacker(self.fmodel, images, labels, epsilons=[self.epsilon])
        return adv_imgs
    
    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx:int) -> torch.Tensor:
        imgs, labels = batch
        adv_imgs = self.model_step(images=imgs, labels=labels)

        logits = self.forward(adv_imgs)

        return logits
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
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



