from tqdm import tqdm 
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

data_path = "/content/dlnn-project_ia-group_15/data/"  # Path to the data folder
anotation_path= "/content/dlnn-project_ia-group_15/anotations_translated_corrected.pkl" # Path to the pickle file with the words spotted in each image
path_fasttext = ""  # Path to the fasttext model, if fasttext is used must be specified. The code will save the file in the path specified if it is not downloaded previously
                    # In this case is an empty string as we are using "glove-wiki-gigaword-300" embedding

img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"

# data_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/data/"
# anotation_path= "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_keras.pkl"
# img_dir = data_path + "JPEGImages"
# txt_dir = data_path + "ImageSets/0"
# path_fasttext = "/home/xnmaster/Project/cc.en.300.bin"

data_path = r"C:\Users\Joan\Desktop\Deep learning project 2\features\data"
anotation_path= r"C:\Users\Joan\Desktop\Deep learning project 2\dlnn-project_ia-group_15\anotations_translated_corrected.pkl"
img_dir = data_path + r"\JPEGImages"
txt_dir = data_path + r"\ImageSets\0"
path_features = r"C:\Users\Joan\Desktop\Deep_Learning_project\dlnn-project_ia-group_15\features_extracted.pkl"



def create_anotations(dim_w2v = 300, max_n_words = 40, anotation_path = anotation_path, path_fasttext = path_fasttext, approach = "glove"):
    # Read the file with the text anotations
    anotations = pd.read_pickle(anotation_path)

    if approach == "fasttext":
        # Download the fasttext model and load it
        if not os.path.isfile(path_fasttext):
            print("Downloading fasttext model...")
            fasttext.util.download_model('en', if_exists='ignore')
            os.rename("./cc.en.300.bin", path_fasttext)

        w2v = fasttext.load_model(path_fasttext)
    
    else:
        # Download the glove model and load it
        w2v = api.load('glove-wiki-gigaword-300')
        vocab = set(w2v.key_to_index.keys()) # words with embedding in glove

    anotation_vecs = {}
    for i, img_name in tqdm(enumerate(anotations.index)):
        if i % 3000 == 0:
            print("Processed {} images out of {}".format(i, len(anotations.index)))
        
        # Read the list of text anotation of the image
        words_OCR = anotations[anotations.index == img_name].values[0][0]

        # For each image we create a matrix of dim (max_n_words, dim_w2v) with the embeddings of the words
        # and a mask to avoid the padding later
        i = 0
        words = np.zeros((max_n_words, dim_w2v))
        text_mask = np.ones((max_n_words,), dtype=bool)
        for word in words_OCR:
            if i < max_n_words:
                if len(word) > 2:  # Remove words with less than 2 characters
                    
                    if approach == "glove": 
                        if (word.lower() in vocab): # We need to check if the word is in the vocabulary of glove
                            words[i,:] = w2v[word.lower()] # Pass the word trohugh the embedding
                            text_mask[i] = True # Set the mask to false, to take into account this word
                            i += 1
                    
                    else:
                        words[i,:] = w2v.get_word_vector(word.lower()) # Pass the word trohugh the embedding
                        text_mask[i] = True # Set the mask to false, to take into account this word
                        i += 1
            else:
                break
            
        # Save the matrix and the mask in a dictionary with the image name as key
        anotation_vecs[img_name] = (words, text_mask)

    return anotation_vecs

def make_loader(dataset, batch_size, shuffle=False):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size, 
                        shuffle=shuffle,
                        pin_memory=True, num_workers=4)
    
    # returns the loader for the dataset given
    return loader

def init_parameters(model):
    # Initialize the parameters of the model 
    for name, w in model.named_parameters():
        if ("feature_extractor" not in name) and ("norm" not in name):
            if ("weight" in name):
                nn.init.xavier_normal_(w)
            if "bias" in name:
                nn.init.ones_(w)

def make(config, device="cuda"):
    # Prepare data, create model, optimizer and criterion from config.

    global data_path, anotation_path, img_dir, txt_dir
    
    input_size = 224 # size of the model input images

    data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # get the anotations (text) of images
    print("Creating anotations...")
    anotations = create_anotations(dim_w2v = 300, max_n_words = 50, anotation_path = anotation_path, path_fasttext = path_fasttext, approach = "glove")

    # Load the labels of the images and split them into train, test and validation
    train_img_names, y_train, test_img_names, y_test, val_img_names, y_val = load_labels_and_split(txt_dir)
    
    # Creating the datasets and the loaders for the train, test and validation
    train_dataset = Dataset_ConText(img_dir, train_img_names, y_train, anotations, transform=data_transforms_train, augment=augment_data)
    test_dataset = Dataset_ConText(img_dir, test_img_names, y_test, anotations, transform=data_transforms_test)
    val_dataset = Dataset_ConText(img_dir, val_img_names, y_val, anotations, transform=data_transforms_test)

    if type(config) == dict: # We check if the config is a dictionary or a wandb object, this is to load the model also from a dict file, we do this during the testing of the model
        train_loader = make_loader(train_dataset, config["batch_size"], shuffle=True)    
        test_loader = make_loader(test_dataset, config["batch_size_val_test"])
        val_loader = make_loader(val_dataset, config["batch_size_val_test"])
        
        model = Transformer_positional_encoding_not_learned(num_classes=config["classes"], depth_transformer=config["depth"], heads_transformer=config["heads"], dim_fc_transformer=config["fc_transformer"]).to(device)
        
        return model, train_loader, test_loader, val_loader
    
    else: # When training the model, the config is a wandb object
        train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
        test_loader = make_loader(test_dataset, config.batch_size_val_test)
        val_loader = make_loader(val_dataset, config.batch_size_val_test)
        
        model = Transformer_positional_encoding_not_learned(num_classes=config.classes, depth_transformer=config.depth, heads_transformer=config.heads, dim_fc_transformer=config.fc_transformer, drop=config.dropout).to(device)
        
        # Initialize the parameters of the model, It is commented because it gave worse results
        # init_parameters(model) 

        # Make the loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        return model, criterion, optimizer, train_loader, test_loader, val_loader
    
def load_labels_and_split(path_sets, random_state=42):
    # Load the labels of the images and split them into train, test and validation

    list_sets = os.listdir(path_sets)
    images_class = {}

    # Get the list of names of images for each class
    for set_filename in list_sets:
        if "_" in set_filename: # to avoid the all.txt file
            with open(path_sets + "/" + set_filename, "r") as file:
                file_text = file.read().splitlines()
            images_class[set_filename] = [row.split()[0] for row in file_text if row.split()[1] == "1"]
    

    # Join all the images and labels in a single list and split them into train, test and validation
    all_img_names = []
    all_y = []
    for key in images_class.keys():
        for image in images_class[key]:
            all_img_names.append(image + ".jpg")
            all_y.append(key.split("_")[0])

    # Split the data into train, test and validation, with stratified sampling to have a similar distribution of  each class in all the sets
    # The partition is 70% train, 15% validation and 15% test
    train_img_names, test_img_names, y_train, y_test = train_test_split(all_img_names, all_y, test_size=0.3, stratify=all_y, random_state=random_state)
    test_img_names, val_img_names, y_test, y_val = train_test_split(test_img_names, y_test, test_size=0.5, stratify=y_test, random_state=random_state)
    return train_img_names, y_train, test_img_names, y_test, val_img_names, y_val

class Dataset_ConText(Dataset):
    # Dataset class for the ConText dataset
    def __init__(self, img_dir, img_list,labels_list, anotations, transform=None, augment=False):
        self.img_dir = img_dir
        self.img_list = img_list
        self.labels_list = labels_list
        self.transform = transform
        self.anotations = anotations
        self.augment = augment

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        
        if self.augment: # if augment is set to True, then the images are grabbed from the augmented folder
            img  = Image.open(os.path.join(self.img_dir+"_augmented", img_name)).convert('RGB')
            
            if len(img_name.split("-")) > 1:
                img_name = img_name.split("-")[0] + ".jpg"
            else:
                raise Exception("The image name is not correct", img_name)

        else:
            img  = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB') # convert to RGB because some images are in grayscale

        
        label = self.labels_list[idx]
        words = self.anotations[img_name][0]
        text_mask = self.anotations[img_name][1]

        # apply the transformations to the image
        if self.transform:
            img = self.transform(img)

        # for each idx we return the image, the label, the words and the text mask
        return (int(label)-1), img, np.array(words), text_mask


# NOT OPTIMIZING CNN HELPER FUNCTIONS AND CLASSES

# A dataset class that gives the img, label, and the image name
# We used it to extract the features from the images using a CNN, as a feature extractor
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

        # return the image and the label
        return (int(label)-1), img, img_name

# This is a dataset class for the ConText dataset, but in this case it is used when the features from the images are already extracted
class Dataset_ConText_Features(Dataset):
    # since the features from the image are already extracted, we can use this class to load them
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

        # Preprocess the words
        # As this is based in a previous version in which we processeed the words in the dataset class
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
    """ 
    The difference with the main "make" function is that here we already extracted the features from the images 
    using the CNN so we just need to load them and create the datasets and loaders 
    """
    
    global data_path, anotation_path, img_dir, txt_dir, path_features
    
    w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding

    ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    # Load the labels of the images and split them into train, test and validation
    with open(path_features, "rb") as f:
        data = pickle.load(f)
    
    # Creating the datasets and the loaders for the train, test and validation
    train_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["train"], anotations=ocr_data, embed=w2v)
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)

    test_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["test"], anotations=ocr_data, embed=w2v)
    test_loader = make_loader(test_dataset, config.batch_size_val_test)

    val_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["val"], anotations=ocr_data, embed=w2v)
    val_loader = make_loader(val_dataset, config.batch_size_val_test)
    
    # Make the model
    model = Transformer_without_extracting_features(num_classes=config.classes, depth_transformer=config.depth, heads_transformer=config.heads, dim_fc_transformer=config.fc_transformer).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, criterion, optimizer, train_loader, test_loader, val_loader

def make_features_test(config, device="cuda"):
    """
    This function is used to load the dataloaders, and the model for the test.
    We discarded the idea of extracting features of the images before hand. So, this function is a past version,
    for this reason it is not incorporated in the "make_features" function, as in the first make of this file.
    """
    # Make the data and model
    global data_path, anotation_path, img_dir, txt_dir, path_features
    w2v = api.load('glove-wiki-gigaword-300') # Initialize the embeding

    ocr_data = pd.read_pickle(anotation_path) # Open the data with the data of the OCR
    
    # Load the labels of the images and split them into train, test and validation
    with open(path_features, "rb") as f:
        data = pickle.load(f)
    
    # Creating the datasets and the loaders for the train, test and validation
    train_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["train"], anotations=ocr_data, embed=w2v)
    train_loader = make_loader(train_dataset, config["batch_size"], shuffle=True)

    test_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["test"], anotations=ocr_data, embed=w2v)
    test_loader = make_loader(test_dataset, config["batch_size_val_test"])

    val_dataset = Dataset_ConText_Features(img_dir=img_dir, data=data["val"], anotations=ocr_data, embed=w2v)
    val_loader = make_loader(val_dataset, config["batch_size_val_test"])
    
    model = Transformer_without_extracting_features(num_classes=config["classes"], depth_transformer=config["depth"], heads_transformer=config["heads"], dim_fc_transformer=config["fc_transformer"]).to(device)

    return model, train_loader, test_loader, val_loader
    