import torch
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch import Trainer
from lightning import seed_everything, Callback, LightningDataModule, LightningModule, Trainer
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Any, Dict, List, Optional, Tuple
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import lightning as L
import torch.nn.functional as F
import hydra
import rootutils
from torchattacks import FGSM
from torchvision.utils import make_grid

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def inference(cfg: DictConfig):
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    
    model_class = hydra.utils.get_class(cfg.model._target_)  
    model = model_class.load_from_checkpoint(cfg.ckpt_path)  


    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    
    attack = FGSM(model.classifier, eps=cfg.test.attack_eps)

    
    accuracy = run_attack(model, logger, datamodule, attack)
    logger[0].log_metrics({"infer/acc": accuracy})

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

def best_reconstruction(model, logger, x, y, n_classes):
    """Identify best reconstruction from PuVAE."""
    batch_size = x.shape[0]
 
    images = x.repeat_interleave(n_classes, dim=0)
    labels = torch.eye(n_classes, device=x.device).repeat(batch_size, 1)

    reconstructions, loss, rc_loss, kl_loss = model(images, labels)
    errors = F.mse_loss(reconstructions, images, reduction='none').mean(dim=[2, 3])
    errors = errors.view(batch_size, n_classes)

    best_idxs = errors.argmin(dim=1) + torch.arange(0, batch_size, dtype=torch.int64, device=errors.device) * n_classes 
    best_reconstructions = reconstructions[best_idxs]

    return best_reconstructions, errors.min(dim=1).values


def run_attack(model, logger, datamodule, attack):
    
    correct, total = 0, 0
    n_classes = datamodule.num_classes

    ori_ls, adv_ls, recon_ls = [], [], []
    label_captions, pred_captions = [], []
    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        x, y = batch
        x.requires_grad = True
        labels = torch.argmax(y, dim=1)
        
        
        adv_examples = attack(inputs=x, labels=labels)
        
        
        with torch.no_grad():
            reconstruction, loss, rc_loss, kl_loss = model(adv_examples, y)
            best_reconstructions, errors = best_reconstruction(model, logger, adv_examples, y, n_classes)
            preds = model.classifier(best_reconstructions)
            preds = torch.argmax(preds, dim=1)
            
            correct += (preds == labels).sum().item()
            total += y.size(0)
            

            for i in range(min(len(adv_examples), 5)):  # Log up to 5 examples per batch
                adv_image = adv_examples[i].detach().cpu()
                recon_image = reconstruction[i].detach().cpu()
                original_image = x[i].detach().cpu()

                ori_ls.append(original_image)
                adv_ls.append(adv_image)
                recon_ls.append(recon_image)
                label_captions.append(f"True: {labels[i].item()}")
                pred_captions.append(f"Pred: {preds[i].item()}")

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(datamodule.test_dataloader()):
            
            ori_grid = make_grid(ori_ls[:50], nrow=10, normalize=True)
            adv_grid = make_grid(adv_ls[:50], nrow=10, normalize=True)
            recon_grid = make_grid(recon_ls[:50], nrow=10, normalize=True)

            
            logger[0].log_image(
                key=f"infer/batch",
                images=[ori_grid, adv_grid, recon_grid],
                caption=[
                    f"Original Images: {', '.join(label_captions[:50])}",
                    f"Adversarial Examples: {', '.join(pred_captions[:50])}",
                    f"Reconstructed Images: {', '.join(pred_captions[:50])}",
                ]
            )

            
            ori_ls, adv_ls, recon_ls = [], [], []
            label_captions, pred_captions = [], []
                            

    accuracy = 100 * correct / total
    return accuracy


@hydra.main(version_base=None, config_path="/mnt/banana/student/ptrang/PuVAE/configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    inference(cfg)
    
if __name__ == "__main__":
    main()