<div align="center">

# Purifying Variational AutoEncoder

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/pdf/1903.00585)

</div>

## Description

The Purifying Variational Autoencoder (PuVAE) is used to handle adversarial inputs and noisy data. It combines the representational power of Variational Autoencoders (VAEs) with purification mechanisms to improve resilience and accuracy in downstream tasks such as classification.

![PuVAE Architecture](/notebooks/architecture.png)

## Experiments

I'm so busy recently so hope I can remember this and update someday :>>

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/Page0526/PuVAE
cd PuVAE

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/Page0526/PuVAE
cd PuVAE

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=100 data.batch_size=64
```

To run inference

```bash
python src/inference.py
```
#### Diary

Take me a lot of try-hard to implement this :<. Cried a lot and also learned a lot :))). Hope this repo can help you in some way. OMG, someday my future self will read this and wonder why my code was (is) so ugly :)))).

[Here is our VietNamese report of this paper (Will update soon!)](https://docs.example.com)
