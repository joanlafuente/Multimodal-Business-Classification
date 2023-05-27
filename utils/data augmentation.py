# DATA AUGMENTATION 

# read all the images in the folder apply x transformations and save them in a new folder
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


data_path = "C:/Users/Maria/OneDrive - UAB/Documentos/2º de IA/NN and Deep Learning/dlnn-project_ia-group_15/data/"
anotation_path= r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\dlnn-project_ia-group_15\anotations_keras.pkl"
img_dir = data_path + "JPEGImages"
new_img_dir = data_path + "JPEGImages_augmented"
txt_dir = data_path + "ImageSets/0"
path_model = r"C:\Users\Maria\Downloads\glove_transformer_depth_4_head_n_8_drop_0_4_mobilnet_batch_120.pkl"

input_size = 224

data_transforms_train = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(236, interpolation=torchvision.transforms.InterpolationMode.BICUBIC), #the interpolation is for the case that the image is not square
        torchvision.transforms.RandomRotation(15),
        # torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=1), 
        torchvision.transforms.RandomResizedCrop(input_size, scale=(0.4, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.GaussianBlur(5),
        torchvision.transforms.RandomEqualize(0.2),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#create new folder
if not os.path.exists(new_img_dir):
    os.makedirs(new_img_dir)


num_transforms = 5 # number of final images per original image

list_imgs = os.listdir(img_dir)
for n, img_name in tqdm(enumerate(list_imgs)):
    if n % 100 == 0:
        print("Transformed {} out of {}".format(n, len(list_imgs)))

    img_path = img_dir + "/" + img_name
    img = Image.open(img_path)
    img.save(new_img_dir + "/" + img_name.split(".")[0] + "-" + str(0) + ".jpg")
    
    for i in range(1, num_transforms):
        img_transformed = data_transforms_train(img)
        img_transformed.save(new_img_dir + "/" + img_name.split(".")[0] + "-" + str(i) + ".jpg") #why TypeError: split() takes 1 positional argument but 2 were given