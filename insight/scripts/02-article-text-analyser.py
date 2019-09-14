# -*- coding: utf-8 -*-

# Import packages
import pandas as pd # matrices
#from pandas.plotting import scatter_matrix
import numpy as np # matrices
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
sns.set(style="ticks", color_codes=True, font_scale=1)
import os # file paths
import nltk # NLP
#from nltk.corpus import stopwords
import string

class ArticleTextAnalyser:
    """ Process the TEXT of an article to extract some secondary statistics """
    
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

        # Call functions to analyse text at different levels of granularity
        self.n_sentences, self.a_sentence_length = self.analyse_sentences()
        self.n_words, self.u_words, self.a_word_length = self.analyse_words()
        
        
    def analyse_sentences(self):
        """  Analyse text on the sentence level. """  
        n_sentences = len(self.sentences)
        a_sentence_length = len(self.words)/len(self.sentences)
        return n_sentences, a_sentence_length
    
    def analyse_words(self):
        """  Analyse text on the word level. """     

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
        
        return n_words, u_words, a_word_length
    
#    def text_metrics(self):
#        """ Combine and output analysed text. """   
#        text_metrics = pd.DataFrame([self.article_sentences+self.article_words], 
#                                      columns=["n_sentences",
#                                               "a_sentence_length",
#                                               "n_words",
#                                               "u_words",
#                                               "a_word_length"])
#        return text_metrics

class ArticleCodeAnalyser:
    """ Process the codeblocks of an article to extract some secondary statistics """
    
    def __init__(self,postId, code):
        """ 
        Initialize class. Receive article text as input, split it up into 
        words and sentences for subfunctions to process further.
        
        Arguments:
            self - Internal class references
            postId - Unique identifier. Used to confirm the metrics for a given 
                post are being added to the correct row when merging dataframes 
                later.
            code - Single string variable containing all the code blocks.
            
        Functions:
            detected_language - Analyse the text block to determine which 
                coding language was used. Returns a numerical label.
        """          
        # Set up initial variables, split into different levels 
        # (article/sentence/text) for further analysis.
        self.code = code
        self.postId = postId
        self.line = self.code.split('\n')
        self.words = self.code.split(' ')

        # Analyse code block
        self.detected_language = self.estimate_language()    
       
    def estimate_language(self):
        """  Determine which coding language the article belongs to. """               
        # Initial variable to be overwritten if language recognized
        detected_language = 0
        
        # Set up language keywords
        python_keywords = ["import","#"]
        javascript_keywords = ["var","//"]
        cmdline_keywords = ["sudo","apt-get","pip","$"]
        sql_keywords = ["SELECT"]
        
        # Assign language based on presence of keywords
        if any(keyword in self.words for keyword in python_keywords):
            detected_language = 1 # Python
        if any(keyword in self.words for keyword in javascript_keywords):
            detected_language = 2 # Javascript
        if any(keyword in self.words for keyword in cmdline_keywords):
            detected_language = 3 # cmdline
        if any(keyword in self.words for keyword in sql_keywords):
            detected_language = 4 # SQL
            
        return detected_language

        
# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/filtereddata.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_filtered = pd.read_csv(filename,
                                index_col = "postId")

# Analyse code
articles_filtered = articles_filtered.assign(codingLanguage = np.zeros(len(articles_filtered)))

for i in range(0,len(articles_filtered)):
    if np.remainder(i,500) == 0:
        print("Processing article code, checkpoint: >= {}".format(i))
    analysed_code = ArticleCodeAnalyser(
            articles_filtered["postId2"].iloc[i],
            articles_filtered["codeBlock"].iloc[i])
    
    articles_filtered.at[analysed_code.postId,"codingLanguage"] = analysed_code.detected_language


# Analyse text
articles_filtered = articles_filtered.assign(n_sentences = np.zeros(len(articles_filtered)),
                                             a_sentence_length = np.zeros(len(articles_filtered)),
                                             n_words = np.zeros(len(articles_filtered)),
                                             u_words = np.zeros(len(articles_filtered)),
                                             a_word_length = np.zeros(len(articles_filtered))
                                             )

for i in range(0,len(articles_filtered)):
    if np.remainder(i,500) == 0:
        print("Processing article text, checkpoint: >= {}".format(i))
    
    analysed_text = ArticleTextAnalyser(
            articles_filtered["postId2"].iloc[i],
            articles_filtered["text"].iloc[i])
    
    articles_filtered.at[analysed_text.postId,"n_sentences"] = analysed_text.n_sentences
    articles_filtered.at[analysed_text.postId,"a_sentence_length"] = analysed_text.a_sentence_length
    articles_filtered.at[analysed_text.postId,"n_words"] = analysed_text.n_words
    articles_filtered.at[analysed_text.postId,"u_words"] = analysed_text.u_words
    articles_filtered.at[analysed_text.postId,"a_word_length"] = analysed_text.a_word_length   


#print("Python-containing articles: {}".format(python_count))    
#print("Javascript-containing articles: {}".format(javascript_count))    
#print("cmdline-containing articles: {}".format(cmdline_count))    
#print("SQL-containing articles: {}".format(sql_count))    
#discarded_articles = len(articles_filtered) - python_count - javascript_count - cmdline_count - sql_count
#print("Discarded articles: {} of {}".format(discarded_articles,len(articles_filtered)))

bins_words = np.arange(0,5001,250)
bins_sentencelengths = np.arange(0,50,2.5)

# Distribution plots of new metrics
fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,3,1)
sns.distplot(articles_filtered["codingLanguage"],bins=5,kde=False)
ax1.set_xlabel("Language Code")
ax1.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5])
ax1.set_xticklabels(["None","Python","Javascript","cmdline","SQL"],rotation=45)
ax1.xaxis.set_tick_params(length=0)

ax2 = fig.add_subplot(1,3,2)
sns.distplot(articles_filtered["n_words"],kde=False,bins=bins_words)
ax2.set_xlabel("Word Count")

ax3 = fig.add_subplot(1,3,3)
sns.distplot(articles_filtered["a_sentence_length"],kde=False,bins=bins_sentencelengths)
ax3.set_xlabel("Average Sentence Length")

plt.subplots_adjust(wspace=0.2,hspace=0.5,bottom=0.2)
    