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
import gensim.downloader as api

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


def make_prev(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConTextTransformer(config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer




def make(config, device="cuda"):
    # Make the data and model
    data_path = "C:/Users/Joan/Desktop/Deep_Learning_project/features/data/"
    img_dir = data_path + "JPEGImages"
    txt_dir = data_path + "ImageSets/0"
    anotation_path= "C:/Users/Joan/Desktop/Deep_Learning_project/dlnn-project_ia-group_15/anotations.pkl"
    input_size = 224
    data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding

    ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    # Load the labels of the images and split them into train, test and validation
    train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(txt_dir)
    # Creating the datasets and the loaders for the train, test and validation
    # Train
    train_dataset = Dataset_ConText(img_dir, train_img_names, y_train, ocr_data, w2v, transform=data_transforms_train)
    train_loader = make_loader(train_dataset, config.batch_size)
    # Test
    test_dataset = Dataset_ConText(img_dir, test_img_names, y_test, ocr_data, w2v, transform=data_transforms_test)
    test_loader = make_loader(test_dataset, config.batch_size_val_test)
    # Validation
    val_dataset = Dataset_ConText(img_dir, val_img_names, y_val, ocr_data, w2v, transform=data_transforms_test)
    val_loader = make_loader(val_dataset, config.batch_size_val_test)
    
    # Make the model
    model = Transformer(num_classes=config.classes, depth_transformer=3, heads_transformer=5, dim_fc_transformer=300).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, criterion, optimizer, train_loader, test_loader, val_loader
    

# It loads the labels of the images and split them into train, test and validation
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

    
    
class Dataset_ConText(Dataset):
    def __init__(self, img_dir, img_list,labels_list, anotations, embed, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.labels_list = labels_list
        self.transform = transform
        self.anotations = anotations
        self.w2v = embed
        self.dim_w2v = 300
        self.vocab = set(self.w2v.key_to_index.keys())
        self.max_n_words = 40

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        img  = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        label = self.labels_list[idx]
        words_OCR = self.anotations[self.anotations.index == img_name].iloc[0]

        if self.transform:
            img = self.transform(img)

        words = np.zeros((self.max_n_words, self.dim_w2v))
        i = 0
        for word in list(set(words_OCR[0])):
            if len(word) > 2:
                if (word.lower() in self.vocab) and (i < self.max_n_words):
                    # words[i,:] = self.w2v[word.lower()]
                    i += 1
        return (int(label)-1), img, np.array(words)
