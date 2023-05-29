# Business classification
Our objective with this project is to classify within 28 different types of businesses (“restaurant”, “bakery”, “pharmacy”, etc) using images, combining textual and visual features. 


In order to do that we have used a transformer encoder architecture.


## The data used
For this project we have used the ConText dataset. You can find this dataset at the following link: https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html


To be able to run the code correctly it is needed to download the features and images zips. Once the files are downloaded extract the folders of both files. From the extracted files two particular folders will be needed, “JPEGImages” and “data” (Is inside the “Features” folder). “JPEGImages” folder with the images should be added to the “data” folder. (falta que afegeixi el ImageSets, no?) esta dins del data diria


If you have done that, you should be able to execute the “main.py” file and train a model, only changing the paths at the start of the “utils.py” file inside the folder “Utils”. In this file it is commented what each path should have. 


## Repository files 
In this repository you can find: despres canviem l’ordre si fa falta
- The main script “main.py” used to train the model
- The train.py script containing the functions to run the training loop, and to track the train and val loss and accuracy to wandb
- The test.py script with the functions to compute the accuracy in the test set
- The utils folder, which contains the utils script with the functions used to create the dataloaders preparing the images and annotations, - load and split the data and initialize the model, loss and optimizer. 
- The folder containing the model. Inside the “models.py” script you can find 3 different models that were tried during this project, the main one that we use is the (buscar nom) which is explained in the next section
- Different ipynb that are useful for visualizing our model’s performance and results. This notebooks are:
 - predicting_img.ipynb
 - test_trained_model.ipynb
 - visualizing_our_model.ipynb
- Lastly the annotations folder contains all annotations with the ocr of text that were extracted from the images.
 - annotations.pkl using easyocr
 - annotations_keras.pkl using keras
 - annotations_corrected_translated.pkl annotations_keras after spell checker and translator
 - annotations_corrected_translated.pkl like the one above but an improved version  
 




## Model
Our final model is based on the one shown at: https://github.com/lluisgomez/ConTextTransformer/blob/main/ConTextTransformer_inference.ipynb 


In this final model the main changes with respect to the model cited previously are:
We are using a mask for the padding in the words vector. So, we do not take into account the padding in the transformer input
We have also changed the positional encoding, in the implementation cited it was a learnable parameter of the neural network, in our implementation we are using a fixed positional encoding based on sinus and cosinus.
We do optimize the CNN, we have made different tests and always the results doing fine tuning for the CNN were much better. In the test_trained_model.ipynb file you can see a test of a model trained without fine tuning.
We have tested different options for the feature extractor of the images. Such us, conv_next tiny, mobile_net, efficient_net_b0,... We have decided to use conv_next tiny as it is the one that gives us best results.
Finally, we have added a parameter to control dropout inside the transformer encoder and added this same dropout into the MLP head, in order to reduce overfitting.


Related to the model but not inside it are the text tokens. We have extracted them using keras-ocr and easy-ocr but we also tried other libraries like pytesseract or PyOCR. We have decided to use keras-ocr as it gives better results obtaining the text but it has a lot of problems with other libraries so, in the repository there is the code to use both. We recommend using the ones extracted by keras-ocr.


Lastly, to represent the words spotted we have tested fasttext and embedding named “glove-wiki-gigaword-300”. The main difference is that fasttext has representation for all the words and the other does not (since it has a closed vocabulary), but in our test this last one. We think that as it does not have representation of the out of the vocabulary words it may be filtering those less important words. We have also only considered only those words of more than two letters to reduce the words that go to the transformer that do not have any particular meaning. 








## Trying our model
An example of our models predictions in real images can be seen in the predicting_img notebook, here is an snippet of what can be seen in that notebook: 


After giving this image to our model

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_15/assets/28900735/84619f11-4a23-4c41-8e27-a7da109ff65c)

It considers that the most likely classes are: 
MedicalCenter 76.1%
School 21.4%
Motel 1.1%


To know what the model “looked at” to do this prediction we can visualize the gradient map in the first cnn layer.

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_15/assets/28900735/e47d8f94-7567-4d3e-bdd4-91e5cd130547)

Looking at where the CNN has focused we could say that the type of windows and left wall make the model take this decision. As the second more probable class is predicting school, the correct prediction. 


Probably if you ask a person to categorize this image it would give the same response, as it does not have any text nor context to understand it is a school. So considering that, it is a good prediction that school is the second more probable class.











## Contributors
Joan Lafuente Baeza, joan.lafuente@autonoma.cat
Maria Pilligua Costa, maria.pilligua@autonoma.cat


Xarxes Neuronals i Aprenentatge Profund
Grau de Artificial Intelligence,
UAB, 2023



