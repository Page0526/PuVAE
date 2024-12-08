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

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

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
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    # Load DataModule and prepare test dataset
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    # Load the pretrained PuVAE classifier model from checkpoint
    # Instantiate model class from config
    model_class = hydra.utils.get_class(cfg.model._target_)  # Get model class
    model = model_class.load_from_checkpoint(cfg.ckpt_path)  # Load checkpoint


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
    # from IPython import embed; embed()
    # Initialize FGSM attack
    attack = FGSM(model.classifier, eps=cfg.test.attack_eps)

    # Run the adversarial attack and evaluate accuracy
    accuracy = run_attack(model, logger, datamodule, attack)
    # print(f"Final Adversarial Test Accuracy: {accuracy:.2f}%")
    logger[0].log_metrics({"infer/acc": accuracy})

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

def best_reconstruction(model, logger, x, y, n_classes):
    """Identify best reconstruction from PuVAE."""
    batch_size = x.shape[0]
 
    images = x.repeat_interleave(n_classes, dim=0)
    labels = torch.eye(n_classes, device=x.device).repeat(batch_size, 1)

    reconstructions, vae_loss = model(images, labels)
    errors = F.mse_loss(reconstructions, images, reduction='none').mean(dim=[2, 3])
    errors = errors.view(batch_size, n_classes)

    best_idxs = errors.argmin(dim=1) + torch.arange(0, batch_size, dtype=torch.int64, device=errors.device) * n_classes 
    best_reconstructions = reconstructions[best_idxs]

    return best_reconstructions, errors.min(dim=1).values


def run_attack(model, logger, datamodule, attack):
    """Run adversarial attack and evaluate model performance."""
    correct, total = 0, 0
    n_classes = datamodule.num_classes

    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        x, y = batch
        x.requires_grad = True
        labels = torch.argmax(y, dim=1)
        
        # Generate adversarial examples
        adv_examples = attack(inputs=x, labels=labels)
        
        # Perform inference on adversarial examples
        with torch.no_grad():
            reconstruction, _ = model(adv_examples, y)
            best_reconstructions, errors = best_reconstruction(model, logger, adv_examples, y, n_classes)
            preds = model.classifier(best_reconstructions)
            preds = torch.argmax(preds, dim=1)
            
            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += y.size(0)

            # Log adversarial images and reconstruction results
            for i in range(min(len(adv_examples), 5)):  # Log up to 5 examples per batch
                adv_image = adv_examples[i].detach().cpu()
                recon_image = reconstruction[i].detach().cpu()
                original_image = x[i].detach().cpu()
                captions = [f"Label: {labels[i].item()}", f"Prediction: {preds[i].item()}", "Reconstruction"]
                # Log images and predictions
                logger[0].log_image(key=f"infer/batch_{batch_idx}_adv_example_{i}",
                                 images=[original_image, adv_image, recon_image],
                                 caption=captions)
                

    accuracy = 100 * correct / total
    return accuracy


@hydra.main(version_base=None, config_path="/mnt/apple/k66/ptrang/PuVAE/configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    inference(cfg)
    
if __name__ == "__main__":
    main()