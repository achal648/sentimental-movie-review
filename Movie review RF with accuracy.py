# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:01:21 2018

@author: achal
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestClassifier

# Import dataset.
data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
del data["id"]
X = data["review"]
y = data["sentiment"]

#Split dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0, shuffle=False)

def review_to_words(raw_review):
    #Function to convert raw review to processed words.
    #1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    
    #2. Remove non letters
    letters = re.sub("[^a-zA-Z]", " ", review_text)
    
    #3. Convert to lower case and split into iondividual words.
    words = letters.lower().split()
    
    #4. Convert stop words to set
    stops = set(stopwords.words("english"))
    
    #5. Remove stop words
    clean_words = [w for w in words if not w in stops]
    
    #6. Join words back to strings and return result
    return (" ".join(clean_words))
  
#Get number of reviews.
num_reviews = X_train.size

print ("Cleaning and parsing training set movie reviews...\n")

#Initialize empty list to hold reviews.
clean_train_reviews = []

#Loop over each review and clean it.
for i in range(0, num_reviews):
    if( (i+1)%500 == 0):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append( review_to_words( X_train[i]))

print ("creating bag of words\n")
from sklearn.feature_extraction.text import CountVectorizer

#Initialize count Vectorizer
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)

#Convert features to array
train_data_features = train_data_features.toarray()

print ("Training Random forest...\n")
#Initialize Random forest with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

#Fit random forest to train data
forest.fit(train_data_features, y_train)

#Testing data

num_reviews_test = len(X_test)

#Total number of reviews from index 0
total_num_reviews = num_reviews + num_reviews_test

#Clean test data
clean_test_review = []

print ("Cleaning and parsing test set movie reviews...\n")

for i in range(num_reviews, total_num_reviews):
    if((i+1)%500 == 0):
        print ("Review %d of %d\n" % (i+1, total_num_reviews))
    clean_test_review.append(review_to_words(X_test[i]))

   
#Creating bag of words for test data
test_data_features = vectorizer.transform(clean_test_review)
test_data_features = test_data_features.toarray()

#Use random forest to make predictions
result = forest.predict(test_data_features)

print ('Accuracy on test set: {:.2f}'.format(forest.score(test_data_features, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, result)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, result))
