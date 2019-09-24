# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:23:23 2019

@author: Lisa
"""

# -*- coding: utf-8 -*-
# Import packages

# Basics
import os
import pandas as pd
import numpy as np

# NLP
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#nltk.tokenize.WhitespaceTokenizer()
#lemmatizer = nltk.stem.WordNetLemmatizer()
#stemmer = nltk.stem.SnowballStemmer()

# Class definition
class TopicModellingSklearn:
    def __init__(self,text,text_proc_check,min_df,max_df,n_components,random_state):
        """ 
        Initialize class. 
        
        Arguments:
            text - DF column containing text block
            text_proc_check = Boolean controlling whether text is preprocessed
            min_df = Minimum number of articles the word must appear in for the
                word to be considered.
            max_df = Threshold for unique words to considered (drop words 
               appearing too frequently, as in stopwords)
            n_topics = Number of topics to consider
            random_seed = Random seed to use for the modelling
        """
        # Set up internal class variables
        self.text = text
        self.text_proc_check = text_proc_check
        self.min_df = min_df
        self.max_df = max_df
        self.n_components = n_components
        self.random_state = random_state
        
        # Pre-process text if requested
        if self.text_proc_check == True:
            self.text_preprocess()
            
            
        # Fit an LDA model
        self.LDA_model, self.word_frequency, self.vocabulary = self.LDA_model()
    
    def text_preprocess(self):
        """ 
        Arguments:
            text - pandas series where each entry is the unprocessed text of
                a given article.
            
        Outputs:
            text_proc - pandas series where the article text has been processed
                based on the following workflow:
                    (1) Case consistency (all lowercase)
                    (2) Lemmatizing
        """
#        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()
        
        for article in range(0,len(self.text)):            
            # Print a status update
            if np.remainder(article,500) == 0:
                print(" ")
                print("Stemming article {}.".format(article))
                print("Original article: {}".format(self.text[article][0:500]))
                
            stop_words = nltk.corpus.stopwords.words('english')
            stop_words_additional = ['model','training','use']
            stop_words.extend(stop_words_additional)
            
            # Process article text, overwrite original text
            article_text_proc = []
            article_text = self.text[article].split(" ")
            for word in article_text:
                word = word.lower()
                if word not in stop_words:
                    article_text_proc.append(lemmatizer.lemmatize(word))
            self.text[article] = " ".join(article_text_proc)
            
            if np.remainder(article,100) == 0:
                print("Lemmatized article: {}".format(self.text[article][0:500]))
            
    def LDA_model(self):
        """ Fit text to an LDA model """
        word_frequency = CountVectorizer(min_df = self.min_df,
                                         stop_words = 'english')
        vocabulary = word_frequency.fit_transform(
                self.text.values.astype('U'))
        
        LDA = LatentDirichletAllocation(n_components = self.n_components,
                                        random_state = self.random_state)
        LDA_model = LDA.fit(vocabulary)
        
        return LDA_model, word_frequency, vocabulary

# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/articles_python.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_python = pd.read_csv(filename,index_col = "postId")

# sklearn/LDA (unsupervised); text case consistency + lemmatization
model_beta = TopicModellingSklearn(text=articles_python["text"],
                                       text_proc_check=True,
                                       min_df = 3,
                                       max_df = 0.25,
                                       n_components = 5,
                                       random_state = 42)

print('Model beta:')
for i, topic in enumerate(model_beta.LDA_model.components_):
    print('Top words for topic {}:'.format(i))
    print([model_beta.word_frequency.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
     
#model_beta_dict = {0 : 'natural language processing',
#                    1 : 'machine learning',
#                    2 : 'supervised learning',
#                    3 : 'natural language processing',
#                    4 : 'machine learning',
#                    5 : 'supervised learning',
#                    6 : 'neural networks',
#                    7 : 'general data science',
#                    8 : 'neural networks',
#                    9 : 'general data science',
#                    10 : 'unassigned'}
#
#topic_prob_beta = model_beta.LDA_model.transform(model_beta.vocabulary)
#articles_python = articles_python.assign(topic_label_beta = np.zeros(len(articles_python)))
#articles_python['topic_label_beta'] = topic_prob_beta.argmax(axis = 1)
#articles_python['topic_label_beta'] = articles_python['topic_label_beta'].map(model_beta_dict)
#
#for i in range(0,50):
#    print("Article #{}".format(i))
#    print("Title:{}".format(articles_python["title"].iloc[i]))
#    print("Label: {}".format(articles_python["topic_label_beta"].iloc[i]))
#    print('\n')
#    
#model_beta_count = articles_python.groupby(["topic_label_beta"]).size().reset_index(
#        name="counts")
#model_beta_count = model_beta_count.sort_values(["counts"],ascending=False)
#print(model_beta_count.head())
#
#
