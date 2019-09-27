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
from sklearn.ensemble import RandomForestRegressor

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

articles_python = articles_python.sort_values(by=["totalClapCount"],ascending=False)
articles_python = articles_python.dropna()
threshold = 0.75
n_articles = int(np.ceil(threshold*len(articles_python)))

articles_python_popular = articles_python[0:n_articles][:]

X = articles_python_popular[["imageCount",
                             "linksCount",
                             "tagsCount",
                             "codeBlockCount",
                             "total_codelines",
                             "total_commentlines",
                             "ratio_codecomment",
                             "a_code_length",
                             "a_comment_length",
                             "n_sentences",
                             "a_sentence_length",
                             "n_words",
                             "u_words",
                             "a_word_length"]]
y = articles_python_popular["recommends"]

# Split intro training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

regressor = RandomForestRegressor(max_depth=10,
                                  random_state=42,
                                  n_estimators=100)
regressor.fit(X_train,np.array(y_train))

#print(regressor.feature_importances_)

print(" ")
print("Random forest model: ")
print('Score: {0:.4f}'.format(regressor.score(X_test,y_test)))
print(" ")
