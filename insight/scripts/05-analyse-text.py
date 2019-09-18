# Import packages

# Basics
#import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from datetime import datetime

## NLP & Machine Learning
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
import string

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=0.95)

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

class SingleArticleParser:
    def __init__(self,article_raw):
        """
        Overview: Take an unprocessed HTML file and analyse its code (<code>)
            and text-based content(<p> or <title>) in order to calculate some
            summary statistics: title, sentence length, code/comment ratio, etc.
            
        Arguments:
            - article_raw = Single string containing unprocessed HTML code.
        
        Outputs:
            - article_processed = DF that contains a column for the summary
                statistics available from the historical dataset (either in the
                original text or calculated from the raw content; see earlier 
                script 02-filtered-article-analyser.py)
        """
        self.article_raw = article_raw
        
        # Pass imported file to article which splits it into text/code and 
        # assigns basic features (current datetime etc.)        
        self.article_processed = self.article_parser()
        print(self.article_processed)
        
        # Set up variables for functions to process text
        self.postId = self.article_processed["postId"]
        self.articletext = self.article_processed["text"].iloc[0]
        self.sentences = self.articletext.split(".")
        self.words = nltk.word_tokenize(self.articletext)
        self.words = [word for word in self.words if not word in list(string.punctuation)]    
    
        # Analyse text in terms of sentences + words
        self.n_sentences, self.a_sentence_length = self.analyse_sentences()
        self.n_words, self.u_words, self.a_word_length = self.analyse_words()
        
        # Set up variables for functions to process code
        self.code = self.article_processed["codeBlock"].iloc[0]
        self.lines = self.code.split('\\n')
        self.n_lines = len(self.code)
        self.code_words = self.code.split(' ')
        
        # Set up empty variables to be returned by default
        self.total_codelines = float('nan')
        self.total_commentlines = float('nan')
        self.ratio_codecomment = float('nan')
        self.a_code_length = float('nan')
        self.a_comment_length = float('nan')

        # Process code
        self.total_codelines, self.total_commentlines, self.ratio_codecomment, \
        self.a_code_length, self.a_comment_length = self.process_code('#')                                                                                                                           
               
        # Add columns to the dataframe with the processed code/text statistics
        self.article_processed = self.article_processed.assign(
               n_sentences = self.n_sentences,
               a_sentence_length = self.a_sentence_length,
               n_words = self.n_words,
               u_words = self.u_words,
               a_word_length = self.a_word_length,
               total_codelines = self.total_codelines,
               total_commentlines = self.total_commentlines,
               ratio_codecomment = self.ratio_codecomment,
               a_code_length = self.a_code_length,
               a_comment_length = self.a_comment_length)
          
    def article_parser(self):
        """ 
        Take an imported HTML file, and parse it to separat title, text, code.
        
        Arguments:
            - self.article_raw = Unprocessed HTML file
            
        Outputs:
            - article_processed = DF that contains columns for title, text, 
                codeblock, and current datetime.
            
        """
        
        soup = BeautifulSoup(self.article_raw,"html.parser")
        
        title = soup.title.string
        
        text_all = str(' ')
        text_blocks = soup.findAll('p')
        for i in range(0,len(text_blocks)):
            text_all = text_all + str(text_blocks[i].contents)[3:-3] + str(' ')
            
        code_all = str(' ')
        code_blocks = soup.findAll('code')
        for i in range(0,len(code_blocks)):
            code_all = code_all + str(code_blocks[i].contents)[2:-2]
            
        time_present = datetime.now()
        time_string = time_present.strftime("%Y-%m-%d %I:%M:%S %p")
            
        article_processed = pd.DataFrame([{'title' : title,
                                           'text' : text_all,
                                           'codeBlock' : code_all,
                                           'firstPublishedDatetime' : time_string,
                                           'postId' : 'draft_article'}],
                             columns = ["title","text","codeBlock","firstPublishedDatetime","postId"],
                             index = ['draft_article'])
            
        return article_processed       
        
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
        
    def process_code(self,commentchar):
        """
        Overview: Take a codeblock and calculate some secondary statistics.
        
        Inputs: 
            - self.lines = List where each entry is a line from a codeblock.
            - commentchar = Character string which indicates the beginning of
                a comment.
        
        Outputs: 
            - total_codelines = Number of lines containing some code
            - total_commentlines = Number of lines containing a comment
            - ratio_codecomment = Ratio of code/comment
            - a_code_length = Average code line length
            - a_comment_length = Average comment line length
        """
        line_containscode = []
        line_containscomment = []
        line_comment_length = []
        line_code_length = []
        
        # Decide if line contains code, comment, or both
        for line in self.lines:
            # Reset tracking variables
            line_code = 0
            line_comment = 0
            
            if line.find(commentchar) == -1:
                 line_code = 1
            else:
                 comment_start = line.index(commentchar)
                 if comment_start == 0:
                     line_code = 0
                     line_comment = 1
                 elif comment_start > 0:
                     alphanum_check = line[0:comment_start].isalnum()
                     if alphanum_check == 0:
                         line_code = 0
                         line_comment = 1
                     else:
                         line_code = 1
                         line_comment = 1
            line_containscode.append(line_code)
            line_containscomment.append(line_comment)
            
            # Average comment line length
            if line_comment == 1:
                comment_length = len(line) - line.index(commentchar)
                line_comment_length.append(comment_length)                                   
            else:
                comment_length = float('nan')
                line_comment_length.append(comment_length)
                
            # Average code line length
            if line_code == 1:
                if line_comment == 1:
                    code_length = line.index(commentchar) - 1
                    line_code_length.append(code_length)                                   
                else:
                    code_length = len(line)
                    line_code_length.append(code_length)
            else:
                code_length = float('nan')
                line_code_length.append(code_length)
            
        # Calculate some code/comment ratios
        total_codelines = sum(line_containscode)
        total_commentlines = sum(line_containscomment)
        a_code_length = np.nanmean(line_code_length)
        a_comment_length = np.nanmean(line_comment_length)
        
        if total_commentlines > 0:
            ratio_codecomment = total_codelines/total_commentlines
        else:
            ratio_codecomment = 0

        return total_codelines, total_commentlines, ratio_codecomment, \
                a_code_length, a_comment_length
       
# Set up file path to draft article
filedir_draft = os.path.dirname(os.path.realpath('__file__'))
filename_draft = 'draft_article.txt'
filename_draft = os.path.join('../data/raw/'+filename_draft)
filename_draft = os.path.abspath(os.path.realpath(filename_draft))

# Open and import file
article_filename = open(filename_draft,"r")
article_raw = article_filename.read()
article_filename.close()

# Set up file path to Python article dataset 
# (raw text/code processed in 02-filtered-article-analyser.py)
filedir_python = os.path.dirname(os.path.realpath('__file__'))
filename_python = 'articles_python.csv'
filename_python = os.path.join('../data/processed/'+filename_python)
filename_python = os.path.abspath(os.path.realpath(filename_python))

# Open and import file
articles_python = pd.read_csv(filename_python, index_col = "postId")
articles_python["firstPublishedDatetime"] = pd.to_datetime(
        articles_python["firstPublishedDatetime"])

# Analyse draft article
article_draft = SingleArticleParser(article_raw).article_processed
article_draft["firstPublishedDatetime"] = pd.to_datetime(
        article_draft["firstPublishedDatetime"])

# Convert datetimeobjects to hour of post for later plotting
articles_timelist = []
for i in range(0,len(articles_python)):
    articles_timelist.append(
            articles_python["firstPublishedDatetime"].iloc[i].time().hour)
article_draft_time = article_draft["firstPublishedDatetime"].iloc[0].time().hour

# Set up bins of predefined size for later histograms
bins_words = np.arange(0,5001,250)
bins_sentencelengths = np.arange(0,41,2.5)
bins_ratiocodecomment = np.arange(0,31,2.5)
bins_a_comment_length = np.arange(0,101,5)

fig = plt.figure(figsize=(12,7))
gs = fig.add_gridspec(3,5)


ax2 = fig.add_subplot(gs[0,2])
sns.distplot(articles_python["n_sentences"].dropna(),
             kde=False, bins = bins_a_comment_length)
plt.axvline(x = article_draft["n_sentences"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax2.set_xlabel("Sentence Count")

ax2 = fig.add_subplot(gs[0,3])
sns.distplot(articles_python["n_words"].dropna(),
             kde=False, bins = bins_words)
plt.axvline(x = article_draft["n_words"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax2.set_xlabel("Word Count")

ax3 = fig.add_subplot(gs[0,4])
sns.distplot(articles_python["a_sentence_length"].dropna(),
             kde=False, bins = bins_sentencelengths)
plt.axvline(x = article_draft["a_sentence_length"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax3.set_xlabel("Sentence Length (Words)")

ax4 = fig.add_subplot(gs[1,2])
sns.distplot(articles_python["ratio_codecomment"].dropna(),
             kde=False, bins=bins_ratiocodecomment,color="r")
plt.axvline(x = article_draft["ratio_codecomment"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax4.set_xlabel("Code/Comment Ratio")

ax5 = fig.add_subplot(gs[1,3])
sns.distplot(articles_python["a_code_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="r")
plt.axvline(x = article_draft["a_code_length"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax5.set_xlabel("Av. Code Line Length")

ax6 = fig.add_subplot(gs[1,4])
sns.distplot(articles_python["a_comment_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="r")
plt.axvline(x = article_draft["a_comment_length"].values, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax6.set_xlabel("Av. Comment Length")

ax7 = fig.add_subplot(gs[2,2])
sns.distplot(articles_timelist, kde=False, bins=24, color="g")
plt.axvline(x = article_draft_time, 
            ymin = 0, ymax = 1, color = 'r', linewidth = 2)
ax7.set_xlabel("Time of Day")


plt.subplots_adjust(wspace=0.35,hspace=0.4,top=0.9)
    
