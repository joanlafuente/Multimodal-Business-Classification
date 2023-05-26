import numpy as np
import fasttext
import pandas as pd
from tqdm import tqdm 

fasttext.util.download_model('en', if_exists='ignore')  # English
!mv cc.en.300.bin /home/user/fastText/cc.en.300.bin


data_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/data/"
anotation_path= r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\dlnn-project_ia-group_15\anotations_keras.pkl"
img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"
path_fasttext = "./cc.en.300.bin"

# path_new_anotation = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_vecs.pkl"


anotations = pd.read_pickle(anotation_path)
dim_w2v = 300
w2v = fasttext.load_model(path_fasttext)
# vocab = set(w2v.key_to_index.keys()) # Comented when using fasttext
max_n_words = 40

anotation_vecs = {}
for i, img_name in tqdm(enumerate(anotations.index)):
    print(i, len(anotations.index))
    words_OCR = anotations[anotations.index == img_name].values[0][0]

    words = np.zeros((max_n_words, dim_w2v))
    text_mask = np.ones((max_n_words,), dtype=bool)
    i = 0
    for word in list(set(words_OCR)):
        if len(word) > 2:
            # if (word.lower() in vocab) and (i < max_n_words): # Comented when using fasttext
                # words[i,:] = w2v[word.lower()] # Comented when using fasttext
            if i < max_n_words: # Comented when using glove
                words[i,:] = w2v.get_word_vector(word)  # Comented when using glove
                print(words_OCR, words[i,:]) 
                text_mask[i] = False
                i += 1
    if i > 100:
        break
    
        
    anotation_vecs[img_name] = (words, text_mask)

import pickle
pickle.dump(anotation_vecs, open(path_new_anotation, "wb"))
    
    
#open anotation
import pickle
anotation_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_vecs.pkl"
dic_anotations_original = pickle.load(open(anotation_path, "rb"))
# print(dic_anotations.keys())
print(dic_anotations['n04146050_15720.jpg'])


# open anotation_vecs and divide the file of 100mb each
import pickle
import os
from math import ceil
anotation_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_vecs.pkl"
dic_anotations = pickle.load(open(anotation_path, "rb"))

for i in range(ceil(len(dic_anotations.keys())/1000)):
    dic_anotations_copy = dic_anotations.copy()
    dic_anotations_copy = dict(list(dic_anotations_copy.items())[i*1000:(i+1)*1000])

    pickle.dump(dic_anotations_copy, open(anotation_path.split(".")[0] + "_"+ str(i) + ".pkl", "wb"))

    print(os.path.getsize(anotation_path.split(".")[0] + "_"+ str(i) + ".pkl")/1000000)


#open all anotation_vecs and merge them
import pickle
import os

dic_anotations_total = {}
for i in range(25):
    anotation_path = "/home/xnmaster/Project/dlnn-project_ia-group_15-1/anotations_vecs_" + str(i) + ".pkl"
    dic_anotations = pickle.load(open(anotation_path, "rb"))
    if i == 0:
        dic_anotations_total = dic_anotations.copy()
    else:
        dic_anotations_total.update(dic_anotations)

# comprobation that dic_anotations_total is the same as dic_anotations
print(dic_anotations_total.keys() == dic_anotations_original.keys())
