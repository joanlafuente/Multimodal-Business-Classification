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

anotations = pd.read_pickle(anotation_path)
print("loading anotations...")
w2v = api.load('glove-wiki-gigaword-300')
print("loaded w2v...")
vocab = set(w2v.key_to_index.keys())

translator = Translator()

anotation_vecs = {}
correct = True
translate = True

with open("corrected_words_fixed_hauria_de_ser_igual.txt", "w") as f:
    f.write("")

with open("translated_words_fixed.txt", "w") as f:
    f.write("")

for i, img_name in tqdm(enumerate(anotations.index)):
    if i % 100 == 0:
        print("Processed {} images out of {}".format(i, len(anotations.index)))
    
    words_OCR = anotations[anotations.index == img_name].values[0][0]

    i = 0
    words_OCR_processed = []
    for word in list(set(words_OCR)):
        already_processed_added = False
        if len(word) > 2 and word is not None:
            if correct:
                if word.lower() not in vocab:
                    prev_word = word
                    word = str(Word(word).correct())
                    if prev_word != word:
                        with open("corrected_words_fixed_hauria_de_ser_igual.txt", "a") as f:
                                    f.write(prev_word + " --> " + word + "\n")

            if translate:
                prev_word = word
                
                if word.lower() not in vocab:
                    try:
                        word = translator.translate(word, dest='en').text
                        
                        # print(prev_word, word)
                        if prev_word != word:
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

        if word in vocab and not already_processed_added:
            words_OCR_processed.append(word)
    
    #save in a new dict the anotations with the corrected words
    anotation_vecs[img_name] = words_OCR_processed

    with open("anotations_translated_corrected_fixed.pkl", "wb") as f:
        pickle.dump(anotation_vecs, f)



#open anotations_corrected.pkl and save it in a dataframe
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


print(df_anotations.head())

df_anotations.to_pickle("anotations_translated_corrected_fixed.pkl")


# #CHECKING HOW MANY WORDS THAT WHERE TRANSLATED OR CORRECTED WHERE NOT BEFORE IN THE VOCABULARY AND NOW THEY ARE
# from tqdm import tqdm
# import gensim.downloader as api
# import pandas as pd

# w2v = api.load('glove-wiki-gigaword-300')
# vocab = set(w2v.key_to_index.keys())

# with open("corrected_words.txt", "r") as f:
#     lines = f.readlines()

# with open("used_corrected_words.txt", "w") as f:
#     f.write("")

# used_corrected_words = []
# for line in lines:
#     line = line.split(" --> ")
#     prev_word = line[0]
#     processed_word = line[1].split("\n")[0]

#     if prev_word not in vocab and processed_word in vocab:
#         with open("used_corrected_words.txt", "a") as f:
#             f.write(prev_word + " --> " + processed_word + "\n")

#     if prev_word in vocab and processed_word not in vocab:
#         with open("now_not_used_corrected_words.txt", "a") as f:
#             f.write(prev_word + " --> " + processed_word + "\n")


# from tqdm import tqdm
# import pickle
# with open("used_corrected_words.txt", "r") as f:
#     lines = f.readlines()

# with open("corrected_words.txt", "r") as f:
#     lines2 = f.readlines()

# with open("now_not_used_corrected_words.txt", "r") as f:
#     lines3 = f.readlines()

# with open("anotations_translated_corrected.pkl", "rb") as f:
#     anotations = pickle.load(f)

# count = 0
# for v in anotations.values:
#     count += len(v[0])

# print("Number of corrected words used {} out of {} corrected words".format(len(lines), len(lines2)))
# print("Number of corrected words that are now not used {} out of {} corrected words".format(len(lines3), len(lines2)))
# print("Number of words used that are corrected {} out of {} words in the anotations".format(len(lines), count))


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



# PREPROCESS THE ANOTATIONS FROM anotations_translated_corrected.pkl TO HAVE JUST 1 WORD PER ITEM IN THE LIST OF WORDS
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

with open("anotations_translated_corrected_fixed.pkl", "rb") as f:
    anotations = pickle.load(f)

new_anotations_values = []
for v in anotations.values:
    new_list_words = []
    if len(v[0])>1:
        for word in v[0]:
            print(word)
            if (len(word.split(" "))>1):
                for w in word.split(" "):
                    new_list_words.append(w)
                    with open("anotations_translated_corrected_fixed_V2.txt", "a") as f:
                        f.write(word + " --> " + w + "\n")
                    print(word + " --> " + w)
            else:
                new_list_words.append(word)
    
    new_anotations_values.append(new_list_words)

anotations = dict(zip(anotations.keys(), new_anotations_values))

with open("anotations_translated_corrected_fixed_V2.pkl", "wb") as f:
    pickle.dump(anotations, f)