# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:30:05 2019

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
#from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

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
        lemmatizer = WordNetLemmatizer()
        
        for article in range(0,len(self.text)):
            # Print a status update
            if np.remainder(article,100) == 0:
                print("Pre-processing requested. Lemmatizing checkpoint {}.".format(article))
#                print("Original article: {}".format(self.text[article][0:500]))
            
            # Process article text, overwrite original text
            article_text_proc = []
            article_text = self.text[article].split(" ")
            for word in article_text:
                word = word.lower()
                article_text_proc.append(lemmatizer.lemmatize(word))
            self.text[article] = " ".join(article_text_proc)
            
#            if np.remainder(article,100) == 0:
#                print("Lemmatized article: {}".format(self.text[article][0:500]))
            
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
filename = os.path.join('../data/processed/articles_javascript.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_javascript = pd.read_csv(filename,index_col = "postId")

# Fit the text to several LDA models for several different settings

# Model Alpha: sklearn/LDA (unsupervised); No pre-processing of text
model_alpha = TopicModellingSklearn(text=articles_javascript["text"],
                                       text_proc_check=False,
                                       min_df = 3,
                                       max_df = [],
                                       n_components = 3,
                                       random_state = 8)
print('Model alpha:')
for i, topic in enumerate(model_alpha.LDA_model.components_):
    print('Top words for topic {}:'.format(i))
    print([model_alpha.word_frequency.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
    
model_alpha_dict = {0 : 'general data science',
                    1 : 'machine learning',
                    2 : 'model deployment'}

topic_prob_alpha = model_alpha.LDA_model.transform(model_alpha.vocabulary)
articles_javascript = articles_javascript.assign(topic_label_alpha = np.zeros(len(articles_javascript)))
articles_javascript['topic_label_alpha'] = topic_prob_alpha.argmax(axis = 1)
articles_javascript['topic_label_alpha'] = articles_javascript['topic_label_alpha'].map(model_alpha_dict)

for i in range(0,50):
    print("Article #{}".format(i))
    print("Title:{}".format(articles_javascript["title"].iloc[i]))
    print("Label: {}".format(articles_javascript["topic_label_alpha"].iloc[i]))
    print('\n')
    
    
model_alpha_count = articles_javascript.groupby(["topic_label_alpha"]).size().reset_index(
        name="counts")
model_alpha_count = model_alpha_count.sort_values(["counts"],ascending=False)
print(model_alpha_count.head())


# Model Beta: sklearn/LDA (unsupervised); text case consistency + lemmatization
model_beta = TopicModellingSklearn(text=articles_javascript["text"],
                                       text_proc_check=True,
                                       min_df = 3,
                                       max_df = 0.5,
                                       n_components = 3,
                                       random_state = 8)

print('Model beta:')
for i, topic in enumerate(model_beta.LDA_model.components_):
    print('Top words for topic {}:'.format(i))
    print([model_beta.word_frequency.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
     
model_beta_dict = {0 : 'machine learning',
                    1 : 'machine learning',
                    2 : 'machine learning'}

topic_prob_beta = model_beta.LDA_model.transform(model_beta.vocabulary)
articles_javascript = articles_javascript.assign(topic_label_beta = np.zeros(len(articles_javascript)))
articles_javascript['topic_label_beta'] = topic_prob_beta.argmax(axis = 1)
articles_javascript['topic_label_beta'] = articles_javascript['topic_label_beta'].map(model_beta_dict)

for i in range(0,50):
    print("Article #{}".format(i))
    print("Title:{}".format(articles_javascript["title"].iloc[i]))
    print("Label: {}".format(articles_javascript["topic_label_beta"].iloc[i]))
    print('\n')
    
model_beta_count = articles_javascript.groupby(["topic_label_beta"]).size().reset_index(
        name="counts")
model_beta_count = model_beta_count.sort_values(["counts"],ascending=False)
print(model_beta_count.head())


# Plot distributions of the topic categorizations for each model
fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(1,3,1)
sns.barplot(x = model_alpha_count["topic_label_alpha"],
            y = model_alpha_count["counts"])
plt.xticks(rotation=45)
ax1.set_xlabel("Topic")
ax1.set_ylabel("Articles")
        
ax2 = fig.add_subplot(1,3,2)
sns.barplot(x = model_beta_count["topic_label_beta"],
            y = model_beta_count["counts"])
plt.xticks(rotation=45)
ax2.set_xlabel("Topic")
ax2.set_ylabel("Articles")
    
fig.subplots_adjust(bottom=0.2)