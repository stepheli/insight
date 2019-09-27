# -*- coding: utf-8 -*-
# Import packages

# Basics
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

# Machine Learning
from sklearn import svm
from sklearn.model_selection import train_test_split

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/articles_python_topics.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_python = pd.read_csv(filename,index_col = "postId")
articles_python["firstPublishedDatetime"] = pd.to_datetime(
        articles_python["firstPublishedDatetime"])

# Topic modelling dictionar
label_dict = {'natural language processing' : 0,
              'general data science' : 1,
              'neural networks' : 2,
              'machine learning' : 3,
              'time series modelling' : 4,
              'supervised learning' : 5}

# Convert column datatypes
articles_python["firstPublishedDatetime"] = articles_python["firstPublishedDatetime"].apply(
        lambda timestamp : timestamp.time().hour)
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('natural language processing',0) 
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('general data science',1) 
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('neural networks',2) 
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('machine learning',3) 
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('time series modelling',4) 
articles_python["topic_label_alpha"] = articles_python["topic_label_alpha"].replace('supervised learning',5) 


# Drop articles without comments, where a_comment_length and 
# ratio_codecomment are nan
articles_python = articles_python.dropna()

# Separate into X (features) and y (results)
X = articles_python[["ratio_codecomment",
                     "total_codelines",
                         "a_sentence_length",
                         "firstPublishedDatetime",
                         "topic_label_alpha"]]
y = articles_python[["recommends"]]

articles_python_features = pd.concat([X,y],axis=1)
scatter_matrix(articles_python_features)

# Split intro training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8)

reg = svm.SVR(gamma='scale')
reg.fit(X_train,y_train)
print(" ")
print("SVM model: ")
print('Score: {0:.4f}'.format(reg.score(X_test,y_test)))
print(" ")

