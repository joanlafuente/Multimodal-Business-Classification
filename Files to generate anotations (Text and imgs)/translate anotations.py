""" 
This file was used to postprocess the anotations obtained using keras-ocr. It reads the anotations 
for each image and, if there is a word which is not in the vocabulary of glove it uses a spell 
corrector to try to correct the words. 

In case the word is still not in the vocabulary it also tries to translates the words to english, 
using the google translate api, in case there is a translation available.

The corrected and translated words are saved in two different files: corrected_words.txt and
translated_words.txt. To be able to look at the words that were corrected and translated.

The anotations with the corrected and translated words are saved in a pickle file:
anotations_translated_corrected.pkl
"""

""" 
In the first version of this file, the anotations where saved in a list of words, but some of the
words where composed by more than 1 word, and therefore they automatically would not be in the vocabulary. 
To solve that we postprocess again the anotations spliting those words to ensure we have just 1 
word per item in the list of words. This new version is saved in anotations_translated_corrected_fixed.pkl
"""


import pandas as pd
import numpy as np
import pickle
from googletrans import Translator
from tqdm import tqdm # the difference between tqdm and tqdm.auto is that tqdm.auto automatically selects a proper tqdm wrapper depending on your environment. If you are in a Jupyter Notebook environment, tqdm.notebook.tqdm is used. Otherwise, tqdm.std.tqdm is used. 
import gensim.downloader as api # why gensim.downloader could not be resolved? because it is not installed. pip install gensim
from textblob import Word


data_path = "C:/Users/Maria/OneDrive - UAB\Documentos/2º de IA/NN and Deep Learning/dlnn-project_ia-group_15/data/"
anotation_path= r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\dlnn-project_ia-group_15\anotations_keras.pkl"
img_dir = data_path + "JPEGImages"
txt_dir = data_path + "ImageSets/0"
path_fasttext = r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\cc.en.300.bin"

# Read anotations
anotations = pd.read_pickle(anotation_path)

# Load glove
print("loading anotations...")
w2v = api.load('glove-wiki-gigaword-300')
vocab = set(w2v.key_to_index.keys())

# Initialize translator
translator = Translator()

anotation_vecs = {}
correct = True # if True, it will try to correct the words that are not in the vocabulary
translate = True # if True, it will try to translate the words that are not in the vocabulary

# Create empty files to save the corrected and translated words
with open("corrected_words_fixed_hauria_de_ser_igual.txt", "w") as f:
    f.write("")

with open("translated_words_fixed.txt", "w") as f:
    f.write("")

# Iterate over the anotations and correct and translate the words that are not in the vocabulary
for i, img_name in tqdm(enumerate(anotations.index)):
    # This takes a lot of time, so we print the number of images processed to know everything is working
    if i % 100 == 0:
        print("Processed {} images out of {}".format(i, len(anotations.index)))
    
    # Get the list of words of the image
    words_OCR = anotations[anotations.index == img_name].values[0][0]

    # Iterate over the words of the image
    i = 0
    words_OCR_processed = []
    for word in list(set(words_OCR)):
        already_processed_added = False

        # We ignore words with less than 2 characters
        if len(word) > 2 and word is not None:
            if correct:

                # If the word is not in the vocabulary, we try to correct it
                if word.lower() not in vocab:
                    prev_word = word
                    word = str(Word(word).correct())
                    if prev_word != word:

                        # if the word was corrected, we save it in a file
                        with open("corrected_words.txt", "a") as f:
                                    f.write(prev_word + " --> " + word + "\n")

            if translate:
                
                # If the word is not in the vocabulary, we try to translate it
                if word.lower() not in vocab:
                    prev_word = word
                    # if the word is empty the library throws an error, so we deal with it using a try except
                    try:
                        word = translator.translate(word, dest='en').text
                        
                        if prev_word != word:

                            # if the word was translated, we save it in a file
                            # if the word translated is more than one word, we save each word separately
                            if len(word.split(" ")) > 1:
                                for w in word.split(" "):    
                                    with open("translated_words_fixed.txt", "a") as f:
                                        f.write(prev_word + " --> " + w + "\n")
                                    
                                    if w in vocab:
                                        words_OCR_processed.append(w)
                                already_processed_added = True
                            
                            else:
                                with open("translated_words_fixed.txt", "a") as f:
                                    f.write(prev_word + " --> " + word + "\n")
                    except:
                        pass

        # if the word is in the vocabulary after processing it, we add it to the list of words of the image
        if word in vocab and not already_processed_added:
            words_OCR_processed.append(word)
    
    #save in a new dictionary the anotations with the corrected words
    anotation_vecs[img_name] = words_OCR_processed

    #save the dictionary in a pickle file, it is inside the loop to have checkpoints in case of error
    with open("anotations_translated_corrected_fixed.pkl", "wb") as f:
        pickle.dump(anotation_vecs, f)



#open anotations_corrected.pkl and save it in a dataframe to have it in the same format as the original anotations
import pandas as pd
import pickle
with open("anotations_translated_corrected_fixed.pkl", "rb") as f:
    anotations = pickle.load(f)

# iterate over the dict and save it in a dataframe
tokens_imgs2 ={}
for k,v in anotations.items():
    if v == None: 
        tokens_imgs2[k] = []
    else:
        tokens_imgs2[k] = [v]

anotations = tokens_imgs2

df_anotations = pd.DataFrame.from_dict(anotations, orient='index', columns=['text_detected'])

#save the dataframe in a pickle file
df_anotations.to_pickle("anotations_translated_corrected_fixed.pkl")


# #CHECKING HOW MANY WORDS THAT WHERE TRANSLATED OR CORRECTED WHERE NOT BEFORE IN THE VOCABULARY AND NOW THEY ARE
from tqdm import tqdm
import gensim.downloader as api
import pandas as pd

w2v = api.load('glove-wiki-gigaword-300')
vocab = set(w2v.key_to_index.keys())

with open("corrected_words.txt", "r") as f:
    lines = f.readlines()

with open("used_corrected_words.txt", "w") as f:
    f.write("")

# This just creates a file with the words that were corrected and are now in the vocabulary named used_corrected_words.txt
used_corrected_words = []
for line in lines:
    line = line.split(" --> ")
    prev_word = line[0]
    processed_word = line[1].split("\n")[0]

    if prev_word not in vocab and processed_word in vocab:
        with open("used_corrected_words.txt", "a") as f:
            f.write(prev_word + " --> " + processed_word + "\n")


from tqdm import tqdm
import pickle
with open("used_corrected_words.txt", "r") as f:
    lines = f.readlines()

with open("corrected_words.txt", "r") as f:
    lines2 = f.readlines()

with open("anotations_translated_corrected.pkl", "rb") as f:
    anotations = pickle.load(f)

count = 0
for v in anotations.values:
    count += len(v[0])

print("Number of corrected words used {} out of {} corrected words".format(len(lines), len(lines2)))
print("Number of words used that are corrected {} out of {} words in the anotations".format(len(lines), count))



# #CHECKING HOW MANY WORDS THAT WHERE TRANSLATED WHERE NOT BEFORE IN THE VOCABULARY AND NOW THEY ARE
from tqdm import tqdm
import gensim.downloader as api
import pandas as pd
import pickle

w2v = api.load('glove-wiki-gigaword-300')
vocab = set(w2v.key_to_index.keys())

with open("translated_words_fixed.txt", "r") as f:
    lines = f.readlines()

with open("used_translated_words_fixed.txt", "w") as f:
    f.write("")

# This just creates a file with the words that were translated and are now in the vocabulary named used_translated_words.txt
used_translated_words = []
for line in lines:
    line = line.split(" --> ")
    prev_word = line[0]
    processed_word = line[1].split("\n")[0]
    if len(processed_word .split(" "))>1:
        for proc_word in processed_word.split(" "):
            if prev_word not in vocab and proc_word in vocab:
                with open("used_translated_words_fixed.txt", "a") as f:
                    f.write(prev_word + " --> " + proc_word + "\n")

    else:
        if prev_word not in vocab and processed_word in vocab:
            with open("used_translated_words_fixed.txt", "a") as f:
                f.write(prev_word + " --> " + processed_word + "\n")


with open("used_translated_words_fixed.txt", "r") as f:
    lines = f.readlines()

with open("translated_words.txt", "r") as f:
    lines2 = f.readlines()

with open("anotations_translated_corrected_fixed.pkl", "rb") as f:
    anotations = pickle.load(f)

count = 0
for v in anotations.values:
    count += len(v[0])

print("Number of translated words used {} out of {} translated words".format(len(lines), len(lines2)))
print("Number of words used that are translated {} out of {} words in the anotations".format(len(lines), count))