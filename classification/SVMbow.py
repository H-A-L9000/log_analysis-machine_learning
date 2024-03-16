#import necessary libraries 
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import time
import re

#Import Datasets

data_set = pd.read_csv('/home/Halle.Derry/logdata.csv',encoding="latin1")
#test_set = pd.read_csv('/home/Halle.Derry/logdata_test.csv',encoding="latin1") 

print(data_set)

print(data_set["Sentiment"].value_counts())

positives = data_set[(data_set["Sentiment"] == "positive")]
#positives_test = test_set[(test_set["Sentiment"] == "positive")]
#print(positives["Sentiment"].value_counts())

negatives = data_set[(data_set["Sentiment"] == "negative")]
#negatives_test = test_set[(test_set["Sentiment"] == "negative")]
#print(negatives["Sentiment"].value_counts())

neutrals = data_set[(data_set["Sentiment"] == "neutral")]
#neutrals_test = test_set[(test_set["Sentiment"] == "neutral")] 
#print(neutrals["Sentiment"].value_counts())

import warnings as wrn
wrn.filterwarnings('ignore')

negatives["Sentiment"] = 0
#negatives_test["Sentiment"] = 0 

positives["Sentiment"] = 2
#positives_test["Sentiment"] = 2

neutrals["Sentiment"] = 1
#neutrals_test["Sentiment"] = 1

#data = pd.concat([positives, neutrals, negatives], axis = 0) 

#data.reset_index(inplace=True) 
#print(data.head)
#print(len(data))
import random
print(len(data_set))
print("Preview of Log Lines & Sentiment")

for i in range(1,10):
    random_ind = random.randint(0,len(data_set) - 1)
    print(str(data_set["Log"][random_ind]),end="\nLabel: ")
    print(str(data_set["Sentiment"][random_ind]),end="\n\n")

#Preprocess Text
lemma = WordNetLemmatizer()
preprocessed = []
for text in data_set["Log"]:
    #remove punctuation and special characters 
    text = re.sub("[^a-zA-z]", " ", text)
    #text = re.sub(' +', '', text) 

    #convert to lowercase
    text = text.lower()
    #tokenize
    text = nltk.word_tokenize(text)
    print(text)
    #remove stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    preprocessed.append(text)
print(preprocessed)
print("Preview of Log Lines Preprocessed")

for i in range(0,5):
   print(preprocessed[i],end="\n\n")
print("Vectorize Preprocessed Text")

#Create vocabulary and choose max_features which creates a feature matrix out of the most n frequent words across our list
vectorizer = CountVectorizer(max_features=100)
#fit the vocabulary to the text data preprocessed 
vectorizer.fit(preprocessed)
#create bag of words model
BOW = vectorizer.transform(preprocessed)
print(BOW)
vocabulary = vectorizer.get_feature_names_out()
bow_matrix = BOW.toarray()
print("Vocabulary:")
print(vocabulary)
print("\nBOW Matrix:")
print(bow_matrix)
#Lets see our vocabulary after choosing max_features, may play around with this parameter
#print(vectorizer.vocabulary_)
#print(vectorizer.vocabulary_['succeeded'])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(BOW,np.asarray(data_set["Sentiment"]), test_size=0.20)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.svm import SVC
start_time = time.time()

model = SVC()
model.fit(x_train,y_train)

end_time = time.time()
process_time = round(end_time-start_time,2)
print("took {} seconds".format(process_time))

predictions = model.predict(x_test)
print("data used to make predictions")
new_text = []
new_features = vectorizer.transform(new_text)

print(predictions)
new_predictions = model.predict(new_features)
print(new_predictions)
from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy of model is {}%".format(accuracy_score(y_test,predictions)*100))

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test,predictions)
sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of SVM')
print(classification_report(y_test,predictions))
                                                  
