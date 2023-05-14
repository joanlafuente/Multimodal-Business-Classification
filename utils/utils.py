import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np 
from PIL import Image
from models.models import *

def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def load_labels_and_split(path_sets, random_state=42):
    list_sets = os.listdir(path_sets)
    images_class = {}

    # Load the labels of the images and split them into train, test and validation
    for set_filename in list_sets:
        if "_" in set_filename:
            with open(path_sets + "/" + set_filename, "r") as file:
                file_text = file.read().splitlines()
            images_class[set_filename] = [row.split()[0] for row in file_text if row.split()[1] == "1"]
    
    train_img_names = []
    y_train = []
    test_img_names = []
    y_test = []
    for key in images_class.keys():
        if "train" in key:
            for image in images_class[key]:
                train_img_names.append(image + ".jpg")
                y_train.append(key.split("_")[0])
        elif "test" in key:
            for image in images_class[key]:
                test_img_names.append(image + ".jpg")
                y_test.append(key.split("_")[0]) 

    test_img_names, val_img_names, y_test, y_val = train_test_split(test_img_names, y_test, test_size=0.4, stratify=y_test, random_state=random_state)
    return train_img_names, y_train, test_img_names, y_test, val_img_names, y_val

def load_images(img_names, labels, data_dir):
    img_dir = data_dir + "JPEGImages"

    list_img = []
    for img_name in img_names:
        img = Image.open(os.path.join(img_dir, img_name)) 
        list_img.append(np.array(img))

    data = pd.DataFrame()
    data["img"] = list_img
    data["label"] = labels
    data["name"] = img_names
    data.set_index("name", inplace=True)
    data["label"] = data["label"].astype(int)

    return data

def merge_data(imagesAndLabels, ocr_data):
    data = pd.concat([imagesAndLabels, ocr_data], axis=1, join="inner")
    return data

def make_dataframe(data_dir, anotation_path, train=True):
    sets_dir = data_dir + "/ImageSets/0"
    train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(sets_dir)
    ocr_data = pd.read_pickle(anotation_path)
    if train:
        train_data = load_images(train_img_names, y_train, data_dir)
        val_data = load_images(val_img_names, y_val, data_dir)
        train_data = merge_data(train_data, ocr_data)
        val_data = merge_data(val_data, ocr_data)
        return train_data, val_data
    else:
        test_data = load_images(test_img_names, y_test, data_dir)
        test_data = merge_data(test_data, ocr_data)
        return test_data
    