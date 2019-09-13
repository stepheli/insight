# -*- coding: utf-8 -*-

# Import packages
import pandas as pd # matrices
#from pandas.plotting import scatter_matrix
import numpy as np # matrices
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import os # file paths
import nltk # NLP
from nltk.corpus import stopwords
import string

# Define functions
class ArticleTextAnalyser:
    """
    Process the text of an article to extract some secondary statistics 
    (word length, sentence length, unique word count, etc.)
    """
    def __init__(self,postId,article):
        """ 
        Initialize class. Receive article text as input, split it up into 
        words and sentences for subfunctions to process further.
        
        Arguments:
            self - Internal class references
            postId - Unique identifier. Used to confirm the metrics for a given 
                post are being added to the correct row when merging dataframes 
                later.
            article - Single string variable containing all the article 
                text. Does not have to be preprocessed to remove stray 
                formatting characters.
            stopset - List of common connecting words to ignore.
            
        Functions:
            analyse_sentences: Analyse inividual sentences
            analyse_words: Analyse individual words
            article_metrics: Wrap up analysis and construct an output        
                Outputs:
                    article_metrics - DF with the following column order:
                        1 - n_sentences - Number of sentences
                        2 - a_sentence_length - Average number of words in sentence
                        3 - n_words - Word count
                        4 - u_words - Number of unique words
                        5 - a_word_length - Average length of word in characters
        """  
        # Set up initial variables, split into different levels 
        # (article/sentence/text) for further analysis.
        self.postId = postId
        self.articletext = article
        self.sentences = self.articletext.split(".")
        self.words = nltk.word_tokenize(self.articletext)
        self.words = [word for word in self.words if not word in list(string.punctuation)]

        # Analyse text, return output
        self.article_sentences = self.analyse_sentences()  
        self.article_words = self.analyse_words()
        
       
    def analyse_sentences(self):
        """  Analyse text on the sentence level. """
        
        # Number of sentences
        n_sentences = len(self.sentences)
        
        # Average sentence length
        a_sentence_length = len(self.words)/len(self.sentences)
        
        analysed_sentences = [n_sentences,a_sentence_length]
        
        # Return to main class
        return analysed_sentences
    
    def analyse_words(self):
        """  Analyse text on the word level. """
        
        # Number of words
        n_words = len(self.words)
        
        # Number of unique words
        word_repeated = pd.DataFrame(self.words, columns=["word"])
        
        word_repeated = word_repeated.groupby(["word"]).size().reset_index(
            name="counts") # group repeated words
        word_repeated_count = word_repeated.sort_values(
                ["counts"],ascending=False) # count repeated words
        u_words = len(word_repeated_count) # determine number of unique words
        
      
        # Average length of word
        lengths = []
        for i in range(0,len(self.words)):
            lengths.append(len(self.words[i]))
        
        a_word_length = np.mean(lengths)
        
        analysed_words = [n_words, u_words, a_word_length]
                
        return analysed_words
    
    def article_metrics(self):
        """ Combine and output analysed text. """   
        article_metrics = pd.DataFrame([self.article_sentences+self.article_words], 
                                      columns=["n_sentences",
                                               "a_sentence_length",
                                               "n_words",
                                               "u_words",
                                               "a_word_length"])
        return article_metrics


# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/filtereddata.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_filtered = pd.read_csv(filename)

## Analyse text
stopset = set(stopwords.words('english'))

articles_analysed = pd.DataFrame(columns=["n_sentences",
                                          "a_sentence_length",
                                          "n_words",
                                          "u_words",
                                          "a_word_length"])
for i in range(0,len(articles_filtered)):
    print("Now processing: article {} of {}".format(i+1,len(articles_filtered)))
    article_metrics = ArticleTextAnalyser(articles_filtered["postId"].iloc[i],
                                          articles_filtered["text"].iloc[i]).article_metrics()
    articles_analysed = articles_analysed.append(article_metrics)
    
## Merge this with existing article stats
#articles_allstats = pd.concat([articles_filtered, articles_analysed])

# Distribution plots of new metrics
fig = plt.figure(figsize=(18,7))

ax1 = fig.add_subplot(2,3,1)
sns.distplot(articles_analysed["n_sentences"])
ax1.set_xlabel("Sentences")

ax2 = fig.add_subplot(2,3,2)
sns.distplot(articles_analysed["a_sentence_length"])
ax2.set_xlabel("Average sentence length (words)")

ax4 = fig.add_subplot(2,3,4)
sns.distplot(articles_analysed["n_words"])
ax4.set_xlabel("Word Count")

ax5 = fig.add_subplot(2,3,5)
sns.distplot(articles_analysed["u_words"])
ax5.set_xlabel("Unique Words")

ax6 = fig.add_subplot(2,3,6)
sns.distplot(articles_analysed["a_word_length"])
ax6.set_xlabel("Average word length (characters)")

plt.subplots_adjust(wspace=0.3,hspace=0.3)
    
# Plot distribution of analysed text
fig = plt.figure()
sns.pairplot(articles_analysed)
plt.show()