import torch
import torch.nn.functional as F
import numpy as np
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfFastGradientAttack, L2FastGradientAttack, LinfProjectedGradientDescentAttack, L2ProjectedGradientDescentAttack

# Custom loss functions
def categorical_crossentropy_with_logits(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true, reduction="none")

def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true, reduction="none")

def custom_loss(y_true, y_pred):
    loss = torch.mean(sparse_categorical_crossentropy_with_logits(y_true, y_pred))
    return loss

# Custom PGD attack classes
class CustomLossLinfPGDAttack(LinfProjectedGradientDescentAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model, labels):
        def loss_fn(inputs):
            logits = model(inputs)
            loss = self.loss_fn(labels, logits)
            return torch.sum(loss)
        return loss_fn

class CustomLossL2PGDAttack(L2ProjectedGradientDescentAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model, labels):
        def loss_fn(inputs):
            logits = model(inputs)
            loss = self.loss_fn(labels, logits)
            return torch.sum(loss)
        return loss_fn

# Custom FGSM attack class
class CustomLossLinfFGSMAttack(LinfFastGradientAttack):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def get_loss_fn(self, model, labels):
        def loss_fn(inputs):
            logits = model(inputs)
            loss = self.loss_fn(labels, logits)
            return torch.sum(loss)
        return loss_fn
    
def attacker(attack, model, images, labels, batch_size, epsilons, bounds):
    labels = torch.argmax(labels, dim=1)
    # Wrap the model with Foolbox's PyTorchModel if not already wrapped
    fmodel = PyTorchModel(model, bounds=bounds)
    outcomes = {eps: (0, 0) for eps in epsilons}
    all_imgs, success_imgs, robustness, num_advs = [], [], [], []
    num_batches = int(np.ceil(len(images) / batch_size))

    
    for i in range(num_batches):
        last = (i == num_batches - 1)
        batch_images = images[i * batch_size: (i + 1) * batch_size] if not last else images[i * batch_size:]
        batch_labels = labels[i * batch_size: (i + 1) * batch_size] if not last else labels[i * batch_size:]
        
        with torch.enable_grad():
            batch_images.requires_grad = True
            _, imgs, successes = attack(fmodel, batch_images, batch_labels, epsilons=epsilons)
            successes = successes.cpu().numpy()
            

        num_attacks = len(batch_images)
        for j, eps in enumerate(epsilons):
            success_idxs = successes[j] == 1

            if len(all_imgs) <= j:
                all_imgs.append(imgs[j])
                success_imgs.append(imgs[j][success_idxs])
            else:
                all_imgs[j] = torch.cat((all_imgs[j], imgs[j]), dim=0)
                success_imgs[j] = torch.cat((success_imgs[j], imgs[j][success_idxs]), dim=0)

            num_successes = np.count_nonzero(success_idxs)
            outcomes[eps] = tuple(map(sum, zip((num_successes, num_attacks), outcomes[eps])))

    for i, eps in enumerate(epsilons):
        num_successes, num_attacks = outcomes[eps]
        if len(success_imgs[i]) > 0:
            preds = torch.argmax(model(success_imgs[i]), dim=1)
            predicted_advs = torch.sum(preds == 10).item()
            num_advs.append(predicted_advs)
            num_successes -= predicted_advs

        robustness = round(1.0 - num_successes / num_attacks, 3)
        robustness.append(robustness)

    return all_imgs, success_imgs, robustness, num_advs

# PGD Attack Wrapper
def pgd_attack(model, images, labels, norm="linf",
               steps=40, rel_stepsize=0.01/0.3, random_start=True, batch_size=128, epsilons=[0.03, 0.1, 0.3], bounds=(0, 1)):
    
    if norm == "l2":
        attack = L2ProjectedGradientDescentAttack(steps=steps,rel_stepsize= rel_stepsize, random_start=random_start)
    elif norm == "linf":
        attack = LinfProjectedGradientDescentAttack(rel_stepsize=rel_stepsize, random_start=random_start,steps=steps)


    return attacker(attack, model, images, labels, batch_size, epsilons, bounds)

# FGSM Attack Wrapper
def fgsm_attack(model, images, labels,  
                batch_size=128, epsilons=[0.03, 0.1, 0.3], bounds=(0, 1)):
    
    attack = LinfFastGradientAttack()
    return attacker(attack, model, images, labels, batch_size, epsilons, bounds)

