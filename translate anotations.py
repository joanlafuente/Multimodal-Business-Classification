import pandas as pd
import numpy as np
import pickle
from googletrans import Translator
from tqdm.auto import tqdm # the difference between tqdm and tqdm.auto is that tqdm.auto automatically selects a proper tqdm wrapper depending on your environment. If you are in a Jupyter Notebook environment, tqdm.notebook.tqdm is used. Otherwise, tqdm.std.tqdm is used. 
import gensim.downloader as api # why gensim.downloader could not be resolved? because it is not installed. pip install gensim
from textblob import Word


data_path = "C:/Users/Maria/OneDrive - UAB\Documentos/2º de IA/NN and Deep Learning/dlnn-project_ia-group_15/data/"
anotation_path= r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\dlnn-project_ia-group_15\anotations_keras.pkl"
img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"
path_fasttext = r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\cc.en.300.bin"

anotations = pd.read_pickle(anotation_path)
print("loading anotations...")
w2v = api.load('glove-wiki-gigaword-300')
print("loaded w2v...")
vocab = set(w2v.key_to_index.keys())

translator = Translator()

anotation_vecs = {}
correct = True
translate = False

for i, img_name in tqdm(enumerate(anotations.index)):
    if i % 100 == 0:
        print("Processed {} images out of {}".format(i, len(anotations.index)))
    
    words_OCR = anotations[anotations.index == img_name].values[0][0]

    i = 0
    words_OCR_processed = []
    for word in list(set(words_OCR)):
        if len(word) > 2 and word is not None:
            if correct:
                if word.lower() not in vocab:
                    prev_word = word
                    word = str(Word(word).correct())
                    if prev_word != word:
                        with open("corrected_words.txt", "a") as f:
                                    f.write(prev_word + " --> " + word + "\n")

            if translate:
                prev_word = word
                
                if word.lower() not in vocab:
                    try:
                        word = translator.translate(word, dest='en').text
                        
                        # print(prev_word, word)
                        if prev_word != word:
                            with open("translated_words.txt", "a") as f:
                                f.write(prev_word + " --> " + word + "\n")
                    except:
                        pass

        if word in vocab:
            words_OCR_processed.append(word)
    #save in a new dict the anotations with the corrected words
    anotation_vecs[img_name] = words_OCR_processed

    with open("anotations_corrected.pkl", "wb") as f:
        pickle.dump(anotation_vecs, f)