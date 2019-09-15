# -*- coding: utf-8 -*-

# Import packages
import pandas as pd # matrices
import numpy as np # matrices
import matplotlib.pyplot as plt # plotting
import matplotlib.gridspec as gridspec
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
        self.a_code_length = float('nan')
        self.a_comment_length = float('nan')

        # Determine coding language based on keywords, parse it into codes and 
        # comments accordingly
        self.detected_language = self.estimate_language()    
        if self.detected_language == 1: # Python
            self.total_codelines, self.total_commentlines, self.ratio_codecomment, \
            self.a_code_length, self.a_comment_length = self.process_code('#')                                                                                                                           
        if self.detected_language == 2: # Javascript
            self.total_codelines, self.total_commentlines, self.ratio_codecomment, \
            self.a_code_length, self.a_comment_length = self.process_code('//')  
        if self.detected_language == 4: # SQL
            self.total_codelines, self.total_commentlines, self.ratio_codecomment, \
            self.a_code_length, self.a_comment_length = self.process_code('#')  
            
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
                                             a_code_length = np.zeros(len(articles_filtered)),
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
    articles_filtered.at[analysed_code.postId,"a_code_length"] = analysed_code.a_code_length
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

# Dictionary to map labelled language to name
language_dict = {0 : "None",
                 1 : "Python",
                 2 : "Javascript",
                 3 : "cmdline",
                 4 : "SQL"}    
language_count = articles_filtered.groupby(["codingLanguage"]).size().reset_index(
        name="counts")
language_count = language_count.sort_values(["counts"],ascending=False)
language_count["codingLanguage"] = language_count["codingLanguage"].map(language_dict)
print(language_count.head())


# Separate articles with processed code into separate variables
articles_python = articles_filtered[articles_filtered["codingLanguage"] == 1]
articles_javascript = articles_filtered[articles_filtered["codingLanguage"] == 2]
articles_sql = articles_filtered[articles_filtered["codingLanguage"] == 4]

# Export filtered articles to new csv for faster import in later scripts
filedir_output = os.path.dirname(os.path.realpath('__file__'))

filename_python_output = os.path.join('../data/processed/articles_python.csv')
filename_python_output = os.path.abspath(os.path.realpath(filename_python_output))
articles_python.to_csv(path_or_buf=filename_python_output)

filename_javascript_output = os.path.join('../data/processed/articles_javascript.csv')
filename_javascript_output = os.path.abspath(os.path.realpath(filename_javascript_output))
articles_javascript.to_csv(path_or_buf=filename_javascript_output)

filename_sql_output = os.path.join('../data/processed/articles_sql.csv')
filename_sql_output = os.path.abspath(os.path.realpath(filename_sql_output))
articles_sql.to_csv(path_or_buf=filename_sql_output)

# Set up bins of predefined size for later histograms
bins_words = np.arange(0,5001,250)
bins_sentencelengths = np.arange(0,41,2.5)
bins_ratiocodecomment = np.arange(0,31,2.5)
bins_a_comment_length = np.arange(0,101,5)

# Distribution plots of new metrics
fig = plt.figure(figsize=(12,7))
gs = fig.add_gridspec(3,5)

axmain = fig.add_subplot(gs[0:,0:2])
sns.barplot(x=language_count["codingLanguage"],
            y=language_count["counts"])
axmain.set_xlabel("Coding Language")
axmain.set_ylabel("Articles")

ax1 = fig.add_subplot(gs[0,2])
sns.distplot(articles_python["ratio_codecomment"].dropna(),
             kde=False, bins=bins_ratiocodecomment)
ax1.set_xlabel("")

ax2 = fig.add_subplot(gs[0,3])
sns.distplot(articles_python["a_code_length"].dropna(),
             kde=False, bins = bins_a_comment_length)
ax2.set_xlabel("")

ax3 = fig.add_subplot(gs[0,4])
sns.distplot(articles_python["a_comment_length"].dropna(),
             kde=False, bins = bins_a_comment_length)
ax3.set_xlabel("")

ax4 = fig.add_subplot(gs[1,2])
sns.distplot(articles_javascript["ratio_codecomment"].dropna(),
             kde=False, bins=bins_ratiocodecomment,color="r")
ax4.set_xlabel("")

ax5 = fig.add_subplot(gs[1,3])
sns.distplot(articles_javascript["a_code_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="r")
ax5.set_xlabel("")

ax6 = fig.add_subplot(gs[1,4])
sns.distplot(articles_javascript["a_comment_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="r")
ax6.set_xlabel("")

ax7 = fig.add_subplot(gs[2,2])
sns.distplot(articles_sql["ratio_codecomment"].dropna(),
             kde=False, bins=bins_ratiocodecomment,color="m")
ax7.set_xlabel("Code/Comment Ratio")

ax8 = fig.add_subplot(gs[2,3])
sns.distplot(articles_sql["a_code_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="m")
ax8.set_xlabel("Av. Line Length")

ax9 = fig.add_subplot(gs[2,4])
sns.distplot(articles_sql["a_comment_length"].dropna(),
             kde=False, bins = bins_a_comment_length,color="m")
ax9.set_xlabel("Av. Comment Length")

plt.subplots_adjust(wspace=0.35,hspace=0.3)

# Python-based analytics
articles_python_text = articles_python[["a_sentence_length",
                                                       "n_words",
                                                       "totalClapCount"]]
articles_python_code = articles_python[["ratio_codecomment",
                                                       "a_code_length",
                                                       "a_comment_length",
                                                       "totalClapCount"]]

fig = plt.figure(figsize=(6,6))
sns.pairplot(articles_python_text,height=2,aspect=1.5)
plt.show()

fig = plt.figure(figsize=(6,6))
sns.pairplot(articles_python_code,height=2,aspect=1.5)
plt.show()