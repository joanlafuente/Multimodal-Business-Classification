# pip install langdetect googletrans==4.0.0-rc1

import pandas as pd
import numpy as np
import pickle

pip install langdetect googletrans==4.0.0-rc1

from googletrans import Translator

def translate_text(text, detect = False, target_language='en'):
    translator = Translator()
    if text != '' and text != ' ':
        for lang in translator.detect(text):
            print(lang.lang, lang.confidence)
        # translation = translator.translate(text, dest=target_language)
        # translation is a googletrans.models.TRanslated object with attributes:

        # return translation.text
    return text

# Example usage
translator = Translator(service_urls=['translate.google.com'])
ocr_text = "hola"
print(translator.detect(ocr_text))
for lang in translator.detect(ocr_text):
    print(lang.lang, lang.confidence)
translated_text = translate_text(ocr_text)

print("Translated text:", translated_text)
# why is the confidence None?
# because the translation is not confident

anotation_path = r"C:\Users\Maria\OneDrive - UAB\Documentos\2º de IA\NN and Deep Learning\dlnn-project_ia-group_15\anotations_keras.pkl"
anotations = pd.read_pickle(anotation_path)

# print all anotations that are not empty
for i, img_name in enumerate(anotations.index[:100]):
    words_OCR = anotations[anotations.index == img_name].iloc[0]
    if len(words_OCR[0]) > 0:
        print()
        print(i, img_name, words_OCR[0])
        for word in words_OCR[0]:
            translated_text = translate_text(word)
            if word != translated_text:
                print(word, translated_text)

        