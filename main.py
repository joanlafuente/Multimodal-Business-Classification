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

# remove slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]




def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-tes1", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, criterion, optimizer, data_transforms_train = make2(config)
      data_path = "C:/Users/Joan/Desktop/Deep_Learning_project/features/data/ImageSets/0"
      img_dir = "C:/Users/Joan/Desktop/Deep_Learning_project/features/data/JPEGImages"
      anotation_path= r"C:\Users\Joan\Desktop\Deep_Learning_project\dlnn-project_ia-group_15\anotations.pkl"
      train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(data_path)
      ocr_data = pd.read_pickle(anotation_path)
      train_dataset = Dataset_ConText(img_dir, train_img_names, y_train, ocr_data, transform=data_transforms_train)
      train_loader = make_loader(train_dataset, config.batch_size)
      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)
      
      # and test its final performance
      #test_loader = make(config, train=False)
      #test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=1,
        classes=28,
        batch_size=64,
        learning_rate=5e-3,
        dataset="ConText",
        architecture="Transformer")
    model = model_pipeline(config)

