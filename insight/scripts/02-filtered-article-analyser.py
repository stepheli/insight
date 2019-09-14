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
            process_python - For a block of Python code, determine if lines 
                contain code, comment, or both and calculate some secondary 
                statistics.
        """          
        # Set up initial variables, split into different levels 
        # (article/sentence/text) for further analysis.
        self.code = code
        self.postId = postId
        self.lines = self.code.split('\n')
        self.n_lines = len(self.lines)
        self.words = self.code.split(' ')
        
        # Set up empty variables to be returned by default
        self.total_codelines = float('nan')
        self.total_commentlines = float('nan')
        self.ratio_codecomment = float('nan')
        self.a_comment_length = float('nan')

        # Determine coding language based on keywords, parse it into codes and 
        # comments accordingly
        self.detected_language = self.estimate_language()    
        if self.detected_language == 1:
            self.total_codelines, self.total_commentlines, self.ratio_codecomment, self.a_comment_length = self.process_python()
       
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
    
    def process_python(self):
        """
        Overview: Take a Python codeblock and calculate some secondary 
            statistics.
        
        Inputs: 
            - self.lines = List where each entry is a line from a codeblock.
        
        Outputs: 
            - total_codelines = Number of lines containing some code
            - total_commentlines = Number of lines containing a comment
            - ratio_codecomment = Ratio of code/comment
        """
        line_containscode = []
        line_containscomment = []
        line_comment_length = []
        
        # Decide if line contains code, comment, or both
        for line in self.lines:
            # Reset tracking variables
            line_code = 0
            line_comment = 0
            
            if line.find('#') == -1:
                 line_code = 1
            else:
                 comment_start = line.index('#')
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
            
            if line_comment == 1:
                comment_length = len(line) - line.index('#')
                line_comment_length.append(comment_length)
                                        
            else:
                comment_length = float('nan')
                line_comment_length.append(comment_length)
            
        # Calculate some code/comment ratios
        total_codelines = sum(line_containscode)
        total_commentlines = sum(line_containscomment)
        a_comment_length = np.nanmean(line_comment_length)
        
        if total_commentlines > 0:
            ratio_codecomment = total_codelines/total_commentlines
        else:
            ratio_codecomment = 0

        return total_codelines, total_commentlines, ratio_codecomment, a_comment_length
        
    def process_javascript(self):
        """
        Overview: Take a Javascript codeblock and calculate some secondary 
            statistics.
        
        Inputs: 
            - self.lines = List where each entry is a line from a codeblock.
        
        Outputs: 
            - total_codelines = Number of lines containing some code
            - total_commentlines = Number of lines containing a comment
            - ratio_codecomment = Ratio of code/comment
        """
        line_containscode = []
        line_containscomment = []
        line_comment_length = []
        
        # Decide if line contains code, comment, or both
        for line in self.lines:
            # Reset tracking variables
            line_code = 0
            line_comment = 0
            
            if line.find('//') == -1:
                 line_code = 1
            else:
                 comment_start = line.index('//')
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
            
            if line_comment == 1:
                comment_length = len(line) - line.index('//')
                line_comment_length.append(comment_length)
                                        
            else:
                comment_length = float('nan')
                line_comment_length.append(comment_length)
            
        # Calculate some code/comment ratios
        total_codelines = sum(line_containscode)
        total_commentlines = sum(line_containscomment)
        a_comment_length = np.nanmean(line_comment_length)
        
        if total_commentlines > 0:
            ratio_codecomment = total_codelines/total_commentlines
        else:
            ratio_codecomment = 0

        return total_codelines, total_commentlines, ratio_codecomment, a_comment_length
        
    
# Import data
filedir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/processed/filtereddata.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles_filtered = pd.read_csv(filename,
                                index_col = "postId")

# Analyse code
articles_filtered = articles_filtered.assign(codingLanguage = np.zeros(len(articles_filtered)),
                                             total_codelines = np.zeros(len(articles_filtered)),
                                             total_commentlines = np.zeros(len(articles_filtered)),
                                             ratio_codecomment = np.zeros(len(articles_filtered)),
                                             a_comment_length = np.zeros(len(articles_filtered)))

for i in range(0,len(articles_filtered)):
    if np.remainder(i,500) == 0:
        print("Processing article code, checkpoint: >= {}".format(i))
    analysed_code = ArticleCodeAnalyser(
            articles_filtered["postId2"].iloc[i],
            articles_filtered["codeBlock"].iloc[i])
    
    articles_filtered.at[analysed_code.postId,"codingLanguage"] = analysed_code.detected_language
    articles_filtered.at[analysed_code.postId,"total_codelines"] = analysed_code.total_codelines
    articles_filtered.at[analysed_code.postId,"total_commentlines"] = analysed_code.total_commentlines
    articles_filtered.at[analysed_code.postId,"ratio_codecomment"] = analysed_code.ratio_codecomment
    articles_filtered.at[analysed_code.postId,"a_comment_length"] = analysed_code.a_comment_length


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

# Filtered articles further
articles_python = articles_filtered[articles_filtered["codingLanguage"] == 1]


bins_words = np.arange(0,5001,250)
bins_sentencelengths = np.arange(0,41,2.5)
bins_ratiocodecomment = np.arange(0,31,2.5)
bins_a_comment_length = np.arange(0,101,5)

# Distribution plots of new metrics
fig = plt.figure(figsize=(12,7))

ax1 = fig.add_subplot(2,3,1)
sns.distplot(articles_filtered["n_words"].dropna(),
             kde=False,bins=bins_words)
ax1.set_xlabel("Word Count")

ax2 = fig.add_subplot(2,3,2)
sns.distplot(articles_filtered["a_sentence_length"].dropna(),
             kde=False,bins=bins_sentencelengths)
ax2.set_xlabel("Average Sentence Length")

# Planned update: different colour per bar, overlaid histograms in subplots 
# 5 & 6 with matching colors
ax4 = fig.add_subplot(2,3,4)
languageplot = sns.distplot(articles_filtered["codingLanguage"].dropna(),
             bins=5,kde=False)
ax4.set_xlabel("Language")
ax4.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5])
ax4.set_xticklabels(["None","Python","Javascript","cmdline","SQL"],rotation=45)
ax4.xaxis.set_tick_params(length=0)

ax5 = fig.add_subplot(2,3,5)
sns.distplot(articles_filtered["ratio_codecomment"].dropna(),
             kde=False, bins=bins_ratiocodecomment)
ax5.set_xlabel("Code Lines / Comment Lines")

ax6 = fig.add_subplot(2,3,6)
sns.distplot(articles_filtered["a_comment_length"].dropna(),
             kde=False, bins = bins_a_comment_length)
ax6.set_xlabel("Average Comment Length")

plt.subplots_adjust(wspace=0.15,hspace=0.5,bottom=0.2)
    