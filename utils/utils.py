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
import pickle
import fasttext
import fasttext.util

data_path = "/content/dlnn-project_ia-group_15/data/"
anotation_path= "/content/dlnn-project_ia-group_15/anotations_vecs.pkl"
img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"

data_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/data/"
anotation_path= "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_vecs.pkl"
img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"
path_fasttext = "/home/xnmaster/Project/cc.en.300.bin"

# data_path = r"C:\Users\Joan\Desktop\Deep_Learning_project\features\data"
# anotation_path= r"C:\Users\Joan\Desktop\Deep_Learning_project\dlnn-project_ia-group_15\anotations_keras.pkl"
# img_dir = data_path + r"\JPEGImages"
# txt_dir = data_path + r"\ImageSets\0"
# path_features = r"C:\Users\Joan\Desktop\Deep_Learning_project\dlnn-project_ia-group_15\features_extracted.pkl"

#save the model 'en' in path /home/user/
# fasttext.util.download_model('en', if_exists='ignore')  # English
# !mv cc.en.300.bin /home/xnmaster/Project/cc.en.300.bin



def make_loader(dataset, batch_size, shuffle=False):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size, 
                        shuffle=shuffle,
                        pin_memory=True, num_workers=0)
    return loader


def make(config, device="cuda"):
    # Make the data and model
    global data_path, anotation_path, img_dir, txt_dir
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
    # w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding
    # w2v = fasttext.load_model(path_fasttext) # Initialize the embeding

    # ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    anotations = pd.read_pickle(anotation_path) # Open the data with the data of the OCR2vec and masks
    
    # Load the labels of the images and split them into train, test and validation
    train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(txt_dir)
    
    # Creating the datasets and the loaders for the train, test and validation
    train_dataset = Dataset_ConText(img_dir, train_img_names, y_train, anotations, transform=data_transforms_train)
    if type(config) == dict:
        train_loader = make_loader(train_dataset, config["batch_size"], shuffle=True)
    else:
        train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)

    test_dataset = Dataset_ConText(img_dir, test_img_names, y_test, anotations, transform=data_transforms_test)
    if type(config) == dict:
        test_loader = make_loader(test_dataset, config["batch_size_val_test"])
    else:
        test_loader = make_loader(test_dataset, config.batch_size_val_test)
    
    val_dataset = Dataset_ConText(img_dir, val_img_names, y_val, anotations, transform=data_transforms_test)
    if type(config) == dict:
        val_loader = make_loader(val_dataset, config["batch_size_val_test"])
    else:
        test_loader = make_loader(val_dataset, config.batch_size_val_test)
    
    # Make the model
    model = Transformer(num_classes=config.classes, depth_transformer=config.depth, heads_transformer=config.heads, dim_fc_transformer=config.fc_transformer).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, criterion, optimizer, train_loader, test_loader, val_loader
    
# def make_test(config, device="cuda"):
#     # Make the data and model
#     global data_path, anotation_path, img_dir, txt_dir
#     input_size = 224
    
#     data_transforms_train = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
#         torchvision.transforms.RandomResizedCrop(input_size),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     data_transforms_test = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
#             torchvision.transforms.CenterCrop(input_size),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     # w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding
#     w2v = fasttext.load_model(r"C:\Users\Joan\Desktop\Deep_Learning_project\cc.en.300.bin")

#     ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
#     # Load the labels of the images and split them into train, test and validation
#     train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(txt_dir)
#     # Creating the datasets and the loaders for the train, test and validation
#     # Train
#     train_dataset = Dataset_ConText(img_dir, train_img_names, y_train, ocr_data, w2v, transform=data_transforms_train)
#     train_loader = make_loader(train_dataset, config["batch_size"], shuffle=True)
#     # Test
#     test_dataset = Dataset_ConText(img_dir, test_img_names, y_test, ocr_data, w2v, transform=data_transforms_test)
#     test_loader = make_loader(test_dataset, config["batch_size_val_test"])
#     # Validation
#     val_dataset = Dataset_ConText(img_dir, val_img_names, y_val, ocr_data, w2v, transform=data_transforms_test)
#     val_loader = make_loader(val_dataset, config["batch_size_val_test"])
    
#     # Make the model
#     model = Transformer(num_classes=config["classes"], depth_transformer=config["depth"], heads_transformer=config["heads"], dim_fc_transformer=config["fc_transformer"]).to(device)

#     return model, train_loader, test_loader, val_loader
    


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
    
    all_img_names = []
    all_y = []
    for key in images_class.keys():
        for image in images_class[key]:
            all_img_names.append(image + ".jpg")
            all_y.append(key.split("_")[0])

    train_img_names, test_img_names, y_train, y_test = train_test_split(all_img_names, all_y, test_size=0.4, stratify=all_y, random_state=random_state)
    test_img_names, val_img_names, y_test, y_val = train_test_split(test_img_names, y_test, test_size=0.5, stratify=y_test, random_state=random_state)
    return train_img_names, y_train, test_img_names, y_test, val_img_names, y_val

    
    
class Dataset_ConText(Dataset):
    def __init__(self, img_dir, img_list,labels_list, anotations, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.labels_list = labels_list
        self.transform = transform
        self.anotations = anotations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        img  = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        label = self.labels_list[idx]
        words = self.anotations[img_name][0]
        text_mask = self.anotations[img_name][1]

        if self.transform:
            img = self.transform(img)

        return (int(label)-1), img, np.array(words), text_mask
    

# Not optimizing CNN helper functions and clases

class Dataset_imgs(Dataset):
    def __init__(self, img_dir, img_list, labels_list, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.labels_list = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        img  = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        label = self.labels_list[idx]

        if self.transform:
            img = self.transform(img)
    
        return (int(label)-1), img, img_name

class Dataset_ConText_Features(Dataset):
    def __init__(self, img_dir, data, anotations, embed):
        self.img_dir = img_dir
        self.img_list = data["img_names"]
        self.labels_list = data["labels"]
        self.img_features = data["features"]

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
        img_features = torch.tensor(self.img_features[idx])
        label = self.labels_list[idx]
        words_OCR = self.anotations[self.anotations.index == img_name].iloc[0]


        words = np.zeros((self.max_n_words, self.dim_w2v))
        text_mask = np.ones((self.max_n_words,), dtype=bool)
        i = 0
        for word in list(set(words_OCR[0])):
            if len(word) > 2:
                if (word.lower() in self.vocab) and (i < self.max_n_words):
                    words[i,:] = self.w2v[word.lower()]
                    text_mask[i] = False
                    i += 1
    
        return int(label), img_features, np.array(words), text_mask
    

def make_features(config, device="cuda"):
    # Make the data and model
    global data_path, anotation_path, img_dir, txt_dir
    w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding

    ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    # Load the labels of the images and split them into train, test and validation
    with open(path_features, "rb") as f:
        data = pickle.load(f)
    # Creating the datasets and the loaders for the train, test and validation
    # Train
    train_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["train"], anotations=ocr_data, embed=w2v)
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    # Test
    test_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["test"], anotations=ocr_data, embed=w2v)
    test_loader = make_loader(test_dataset, config.batch_size_val_test)
    # Validation
    val_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["val"], anotations=ocr_data, embed=w2v)
    val_loader = make_loader(val_dataset, config.batch_size_val_test)
    
    # Make the model
    model = Transformer_without_extracting_features(num_classes=config.classes, depth_transformer=config.depth, heads_transformer=config.heads, dim_fc_transformer=config.fc_transformer).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, criterion, optimizer, train_loader, test_loader, val_loader
    


def make_features_test(config, device="cuda"):
    # Make the data and model
    global data_path, anotation_path, img_dir, txt_dir
    w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding

    ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    # Load the labels of the images and split them into train, test and validation
    with open(path_features, "rb") as f:
        data = pickle.load(f)
    # Creating the datasets and the loaders for the train, test and validation
    # Train
    train_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["train"], anotations=ocr_data, embed=w2v)
    train_loader = make_loader(train_dataset, config["batch_size"], shuffle=True)
    # Test
    test_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["test"], anotations=ocr_data, embed=w2v)
    test_loader = make_loader(test_dataset, config["batch_size_val_test"])
    # Validation
    val_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["val"], anotations=ocr_data, embed=w2v)
    val_loader = make_loader(val_dataset, config["batch_size_val_test"])
    
    # Make the model
    model = Transformer_without_extracting_features(num_classes=config["classes"], depth_transformer=config["depth"], heads_transformer=config["heads"], dim_fc_transformer=config["fc_transformer"]).to(device)


    return model, train_loader, test_loader, val_loader
    