# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Walkthrough:
# https://stackabuse.com/python-for-nlp-topic-modeling/

#def text_preprocess(text):
#    """ 
#    Pre-processing a text block for LDA. Steps:
#        (1) Case consistency
#        (2) Stemming
#    
#    Arguments:
#        text - Unseparated text block of raw text
#    Ouputs:
#        text_stemmed - Unseparated text block of stemmed words
#    """
#    stemmer = SnowballStemmer("english")
#    stemmed_text = []
#    words = text.split(' ')     
#    
#    for word in words:
#        word_lower = word.lower() 
#        words_stemmed = stemmer.stem(word_lower)
#        stemmed_text.append(words_stemmed)
#    stemmed_text = " ".join(stemmed_text)
#    return stemmed_text

# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/articles_python.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_python = pd.read_csv(filename,index_col = "postId")


word_frequency = CountVectorizer(min_df = 3,
                                stop_words = 'english')
vocabulary = word_frequency.fit_transform(articles_python["text"].values.astype('U'))

LDA = LatentDirichletAllocation(n_components=10, random_state=8)
LDA.fit(vocabulary)

for i, topic in enumerate(LDA.components_):
    print('Top words for topic {}:'.format(i))
    print([word_frequency.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
    
#for i in range(0,len(articles_python)):
#    articles_python["text_cleaned"].iloc[i] = text_preprocess(articles_python["text"].iloc[i])

## Extract vocabulary
#count_vect = CountVectorizer(max_df=0.50, 
#                             min_df=3, 
#                             stop_words='english')
#doc_term_matrix = count_vect.fit_transform(
#        articles_python["text"].values.astype('U'))
#
## Perform LDA
#LDA = LatentDirichletAllocation(n_components=5, random_state=42)
#LDA.fit(doc_term_matrix)
#   
#for i,topic in enumerate(LDA.components_):
#    print('Top 20 words for topic #{}:.'.format(i))
#    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-20:]])
#    print('\n')
#    
#category_labels = {0 : "machine learning / neural networks",
#                   1 : "machine learning / supervised learning",
#                   2 : "natural language processing",
#                   3 : "machine learning / regressors",
#                   4 : "general data science"}
#   
#topic_values = LDA.transform(doc_term_matrix)
#topic_values.shape
#
#articles_python['topic_label'] = topic_values.argmax(axis=1)
#articles_python.head(50)
#
#articles_python = articles_python.assign(topic_name = np.zeros(len(articles_python)),
#                                             )
#articles_python['topic_name'] = articles_python["topic_label"].map(category_labels)
