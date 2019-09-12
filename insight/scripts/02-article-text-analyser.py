# -*- coding: utf-8 -*-

# Import packages
import pandas as pd # matrices
from pandas.plotting import scatter_matrix
import numpy as np # matrices
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import os # file paths
import nltk # NLP
from nltk.corpus import stopwords

# Define functions
class ArticleTextAnalyser:
    """
    Process the text of an article to extract some secondary statistics 
    (word length, sentence length, unique word count, etc.)
    """
    def __init__(self,articleText,stopset):
        """ 
        Initialize class. Receive article text as input, split it up into 
        words and sentences for subfunctions to process further.
        
        Arguments:
            self - Internal class references
            articletext - Single string variable containing all the article 
                text. Does not have to be preprocessed to remove stray 
                formatting characters.
            stopset - List of common connecting words to ignore.
                
        Outputs:
            articleMetrics - DF with the following column order:
                1 - nSentences - Number of sentences
                2 - aSentenceLength - Average number of words in sentence
                3 - nWords - Word count
                4 - uWords - Number of unique words
                5 - aWordLength - Average length of word in characters
        """
       
        self.articleText = articleText
        self.sentences = articleText.split(".")
        self.words = nltk.word_tokenize(articleText)
        
#        # Note: This stopset includes both common connecting words (the, and, 
#        # of and punctuation. A second generation of filter will be needed to 
#        # remove)
#        self.words = [word for word in self.words if not word in stopset]
        
        # Analyse text based on sub-functions specific to the sentence 
        # or word level
        articleSentences = self.analyseSentences()  
        articleWords = self.analyseWords()
        
        # Combine these metrics into a given list     
        articleMetrics = pd.DataFrame([articleSentences+articleWords], 
                                      columns=["nSentences",
                                               "aSentenceLength",
                                               "nWords",
                                               "uWords",
                                               "aWordLength"])
        return articleMetrics
        
    def analyseSentences(self):
        """  Analyse text on the sentence level. """
        
        # Number of sentences
        nSentences = len(self.sentences)
        
        # Average sentence length
        aSentenceLength = len(self.sentences)/len(self.words)
        
        analysedSentences = [nSentences,aSentenceLength]
        
        # Return to main class
        return analysedSentences
    
    def analyseWords(self):
        """  Analyse text on the word level. """
        
        # Number of words
        nWords = len(self.words)
        
        # Number of unique words
        wordRepeated = pd.DataFrame(self.words, columns=["word"])
        
        wordRepeated = wordRepeated.groupby(["word"]).size().reset_index(
            name="counts") # group repeated words
        wordRepeatedCount = wordRepeated.sort_values(
                ["counts"],ascending=False) # count repeated words
        uWords = len(wordRepeatedCount) # determine number of unique words
                
        # Average length of word
        lengths = []
        for word in wordRepeatedCount.values.tolist():
            lengths.append(len(word))
        aWordLength = int(np.mean(lengths))
        
        analysedWords = [nWords, uWords, aWordLength]
                
        return analysedWords


# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/filtereddata.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_filtered = pd.read_csv(filename)

### Analyse text
#
## Define set of stopwords to be ignored
#stopset = set(stopwords.words('english'))
#
## Testing purposes only: analyse a single article
#analysis = ArticleTextAnalyser(articlesFiltered["text"].iloc[0],stopset)
#
##articleText = articlesFiltered["text"].iloc[0]
##sentences = articleText.split(".")
##words = nltk.word_tokenize(articleText)
##print(len(words))
###words = [word for word in words if not word in stopset]
###print(len(words))
