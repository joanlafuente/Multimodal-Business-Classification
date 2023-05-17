import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-tes1", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, criterion, optimizer, data_transforms_train, train_loader, test_loader, val_loader = make(config)
      # and use them to train the model
      train(model, train_loader, val_loader, criterion, optimizer, config)
      
      # and test its final performance
      test(model, test_loader, save=False)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=1,
        classes=28,
        batch_size=64,
        learning_rate=0.0001,
        patience=5,
        dataset="ConText",
        architecture="Transformer")
    model = model_pipeline(config)

