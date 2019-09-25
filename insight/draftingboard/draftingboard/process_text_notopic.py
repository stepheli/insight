# Data import and handling
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from datetime import datetime
import string
from joblib import dump, load
import pickle

# NLP & Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import LatentDirichletAllocation
import nltk

# Statistics
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=0.95)
	
def master_function(user_input):
    """
    Purpose: Take input text, feed it into subfunctions to process it further, 
    return some things to the Flask script for updating the webpages with.
    	
    Functions called upon:
        - generate_figure: Take input text, generate a plot of the NLP metrics (red 
        line) on a histogram of historically successful articles
        - recommend: Take input text, look for similar articles in the database and 
        return their titles.
        - suggestion1: Generate a suggestion for what to change about the article 
        based on the NLP metrics.
    	
    Classes called upon:
        - SingleArticleParser: Take input text and code, calculate some NLP metrics 
        for average words, ratio of code/comment, etc. This is called early on as 
        the NLP metrics are used in many of the functions described earlier.
    	
    Inputs:
        - user_input = Raw, unformatted text entered by the user.
    	
	Outputs:
        - (no output) Analytics/histogram figure saved to the file path 
        '../static/img/output.png'
        - similar_articles = 3-item list of article titles with the highest similarity 
        to the input text
        - suggestions = String variables containing suggestions for what to fix 
        about the article.
    """
    # Get user input
    article_raw = user_input

    # Read in 2 DFs of previous articles which have already had the same NLP 
    # metrics calculated. articles_popular is used for the histograms 
    # (where you sit relative to popular articles), articles_all is used for 
    # the recommender system (has this already been published before, whether 
    # or not it was popular)
    articles_all = pd.read_pickle('articles_python.pickle')
    articles_popular = pd.read_pickle('articles_python_popular.pickle')
    
    # Feed user input to single article parser (returns a DF that contains NLP 
    # metrics for the article)
    article_draft = SingleArticleParser(article_raw).article_processed
    article_draft["firstPublishedDatetime"] = pd.to_datetime(article_draft["firstPublishedDatetime"])
 
    articles_all = articles_all.append(article_draft)
    article_draft_index = len(articles_all) - 1
    articles_all.loc["draft_article","postId"] = "draft_article"
    
#    # Assign topic
#    article_topic = assign_topic(articles_all)
    
    # Generate a figure that overlays the NLP metric values on a histogram of the historically popular ones
    generate_figure(article_draft, articles_popular)
    	
    # Process text column to construct tfidf_matrix as input for cosine similarity
    topic_frequency = TfidfVectorizer(analyzer = 'word',
									 min_df = 1,
									 stop_words = 'english')
    tfidf_matrix = topic_frequency.fit_transform(articles_all["text"])
    
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix) 
    articles_all = articles_all.reset_index(drop=True)

    indices = pd.Series(articles_all["text"].index)
	
    # Recommend similar articles
    similar_articles = recommend(article_draft_index, indices, cosine_similarity, articles_all)

    # Feed processed draft into the suggestion system
    suggestion1_string = suggestion1(articles_all)
    suggestion2_string = suggestion2(similar_articles)
    suggestion3_string = "Status: debugging."
    suggestions = [suggestion1_string, suggestion2_string, suggestion3_string]
    
    outputs = [similar_articles, suggestions]
    # Return values back to Flask script for display in the output webpage
    return outputs
	
def generate_figure(article_draft, articles_popular):
	"""
	Purpose: Generate a figure which overlays the NLP metrics of the draft article on the histogram of historically successful articles (measured by those in the top percentage of clap count
	
	Inputs:
		- article_draft = 1 row DF containing the characteristics of the draft article (pre-crunched NLP metrics handled by the class SingleArticleParser0
		- articles_popular = DF containing the database of historically successful articles
		
	Outputs:
		- None. (Function does not return anything.) However, it will run a plt.savefig() command with the plotted histograms that the Flask script setup.py will call at one point.
	"""
	article_draft = article_draft
	articles_python = articles_popular
	
	# Convert datetimeobjects to hour of post for later plotting
	articles_timelist = []
	for i in range(0,len(articles_python)):
		articles_timelist.append(
				articles_python["firstPublishedDatetime"].iloc[i].time().hour)
	article_draft_time = article_draft["firstPublishedDatetime"].iloc[0].time().hour
	
	bins_words = np.arange(0,5001,250)
	bins_sentencelengths = np.arange(0,41,2.5)
	bins_ratiocodecomment = np.arange(0,31,2.5)
	bins_a_comment_length = np.arange(0,101,5)

	fig = plt.figure(figsize=(7,7))
	gs = fig.add_gridspec(3,3)

	ax2 = fig.add_subplot(gs[0,0])
	sns.distplot(articles_python["n_sentences"].dropna(),
				 kde=False, bins = bins_a_comment_length)
	plt.axvline(x = article_draft["n_sentences"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax2.set_xlabel("Sentence Count")

	ax2 = fig.add_subplot(gs[0,1])
	sns.distplot(articles_python["n_words"].dropna(),
				 kde=False, bins = bins_words)
	plt.axvline(x = article_draft["n_words"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax2.set_xlabel("Word Count")

	ax3 = fig.add_subplot(gs[0,2])
	sns.distplot(articles_python["a_sentence_length"].dropna(),
				 kde=False, bins = bins_sentencelengths)
	plt.axvline(x = article_draft["a_sentence_length"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax3.set_xlabel("Sentence Length (Words)")

	ax4 = fig.add_subplot(gs[1,0])
	sns.distplot(articles_python["ratio_codecomment"].dropna(),
				 kde=False, bins=bins_ratiocodecomment,color="r")
	plt.axvline(x = article_draft["ratio_codecomment"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax4.set_xlabel("Code/Comment Ratio")

	ax5 = fig.add_subplot(gs[1,1])
	sns.distplot(articles_python["a_code_length"].dropna(),
				 kde=False, bins = bins_a_comment_length,color="r")
	plt.axvline(x = article_draft["a_code_length"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax5.set_xlabel("Av. Code Line Length")

	ax6 = fig.add_subplot(gs[1,2])
	sns.distplot(articles_python["a_comment_length"].dropna(),
				 kde=False, bins = bins_a_comment_length,color="r")
	plt.axvline(x = article_draft["a_comment_length"].values, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax6.set_xlabel("Av. Comment Length")

	ax7 = fig.add_subplot(gs[2,0])
	sns.distplot(articles_timelist, kde=False, bins=24, color="g")
	plt.axvline(x = article_draft_time, 
				ymin = 0, ymax = 1, color = 'r', linewidth = 2)
	ax7.set_xlabel("Time of Day")
	

	plt.subplots_adjust(wspace=0.35,hspace=0.4,top=0.9)
	plt.savefig('static/img/output',bbox_inches='tight')

def recommend(index,indices,method,articles_all):	
    """
    Purpose: Return the titles for articles which are the most similar by text 
    content to the draft.
    	
    Inputs:
        - index = Index of the draft article in the DF articles_all
        - indices = List of all indices in the DF aticles_all
        - method = Approach by which similarity should be calculated
        - articles_all = DF containing the database of articles similarity should 
        be calculated against.
    		
    Outputs:
        - similar_articles = 3-item list containing titles of articles determined 
        to be most similar
    """
    # Calculate similarity score across all articles in database
    id = indices[index]
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        

    # Store list of simscore values for later use in suggestion generation
    similarity_scores_values = []
    for i in range(0,len(similarity_scores)):
        similarity_scores_values.append(similarity_scores[i][1])  
    similarity_scores_mean = np.mean(similarity_scores_values)
    similarity_scores_std = np.std(similarity_scores_values)
  
    similarity_scores_stats = [similarity_scores_mean, similarity_scores_std]
        
    similarity_scores_top = similarity_scores[1:4]  
    articles_index = [i[0] for i in similarity_scores_top]
    similarity_scores_topvalues = []
    for i in range(0,len(similarity_scores_top)):
        similarity_scores_topvalues.append(similarity_scores_top[i][1])
        
        
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return [articles_all['title'].iloc[articles_index], similarity_scores_top, similarity_scores_stats]

def assign_topic(articles_all):
    # Load previously pickled model, set up dictionary for mapping index to topic name
    LDA_model = pickle.load(open('pickled_LDA_model.sav','rb'))
    article_draft = articles_all.loc[["draft_article"]]
    
    model_dict = {0 : 'general machine learning',
                 1 : 'general data science',
                 2 : 'natural language processing',
                 3 : 'natural language processing',
                 4 : 'general data science',
                 5 : 'neural networks',
                 6 : 'regressors, classifiers, and clustering'}
    
    # Pass text of draft article to count vectorizer to prep for fitting
    try:
        word_frequency = CountVectorizer(stop_words='english')
        vocabulary = word_frequency.fit_transform(article_draft["text"])
        
        assigned_topic_index = LDA_model.LDA_model.fit_transform(vocabulary)
        
        # Assign topic with the highest probability, map index to name and return to master_function 
        assigned_topic_index = np.argmax(assigned_topic_index)       
        assigned_topic = model_dict[assigned_topic_index]
        
    except:
        assigned_topic = "unknown"
    
    return assigned_topic
    
def suggestion1(articles_all):
    zscores = pd.DataFrame({'total_codelines' : stats.zscore(articles_all["total_codelines"].iloc[:]),
                            'total_commentlines' : stats.zscore(articles_all["total_commentlines"].iloc[:]),
                            'ratio_codecomment' : stats.zscore(articles_all["ratio_codecomment"].iloc[:]),
                            'a_code_length' : stats.zscore(articles_all["a_code_length"].iloc[:]),
                            'a_comment_length' : stats.zscore(articles_all["a_comment_length"].iloc[:]),
                            'n_sentences' : stats.zscore(articles_all["n_sentences"].iloc[:]),
                            'a_sentence_length' : stats.zscore(articles_all["a_sentence_length"].iloc[:]),
                            'n_words' : stats.zscore(articles_all["n_words"].iloc[:]),
                            'u_words' : stats.zscore(articles_all["u_words"].iloc[:]),
                            'a_word_length' : stats.zscore(articles_all["a_word_length"].iloc[:])
                            },
                            index=articles_all["postId"])
    zscores_abs = zscores.apply(np.absolute, axis=1)
    zscores_draft = [np.argmax(zscores_abs.loc["draft_article"][:]),
                     np.max(zscores_abs.loc["draft_article"][:])]
    
    dict_characteristics = {'total_codelines' : 'total of code lines',
                            'total_commentlines' : 'total of comment lines',
                            'ratio_codecomment' : 'ratio of code/comment lines',
                            'a_code_length' : 'average length of code line',
                            'a_comment_length' : 'avergae length of commentline',
                            'n_sentences' : 'total number of sentences',
                            'a_sentence_length' : 'average words/sentence',
                            'n_words' : 'word count',
                            'u_words' : 'vocabulary size',
                            'a_word_length' : 'average sentence length'}
    char_draft = dict_characteristics.get(zscores_draft[0])
    value_draft = articles_all[zscores_draft[0]].iloc[len(articles_all)-1]
    sign_draft = str(np.sign(zscores.loc["draft_article",str(zscores_draft[0])]))
    mean_all = np.mean(articles_all[zscores_draft[0]])
    
    dict_sign_descript = {'-1.0' : 'lower',
                          '1.0' : 'higher'}
    sign_sugg_descript = dict_sign_descript.get(sign_draft)
    
    dict_sign = {'-1.0' : "increasing",
                 '1.0' : "decreasing"}
    sign_sugg = dict_sign.get(sign_draft)
    
    suggestion1 = "Your " + char_draft + " of " + "{:.2f}".format(value_draft) + \
    " is " + sign_sugg_descript + " than the average of " "{:.2f}".format(mean_all) + \
    ", consider " + sign_sugg + " this for increased clarity."
    
    return suggestion1

def suggestion2(similar_articles): 
    """
    Purpose: Generate a suggestion based on article similarity.
    
    Inputs:
        - similar_articles 
        
    Outputs:
        - string with suggestion to be displayed
    """
    sim_top = similar_articles[1][0][1]
    sim_mean = similar_articles[2][0]
    sim_std = similar_articles[2][1]
        
    if sim_top < 5*sim_mean: # threshold for determining uniqueness.
        output="This content appears unique in the reference database! Great work."
    else:
        output="Take a second look at the articles deemed to be similar to make \
              sure yours is unique."
    
    suggestion2 = "The similarity score between your draft and the next closest article is {a:.2f}, \
    compared to an average of {b:.2f}. {d})".format(a=sim_top,b=sim_mean,d=output)
    
#    suggestion2 = "42: the answer to the question of life, the universe, and everything."
    
    return suggestion2

#def suggestion3(article_topic):
#    """
#    Purpose: Generate a suggestion for tags to add based on the detected topic.
#    """
#    
#    
#    if article_topic == 'general machine learning':
#        keywords = 'machine learning, \
#            artificial intelligence, \
#            scoring, \
#            and confusion matrix.'
#    elif article_topic == 'general data science':
#        keywords = 'data science, \
#            programming, \
#            data visualization \
#            and accuracy.'
#    elif article_topic == 'natural language processing':
#        keywords = 'machine learning \
#            NLP, \
#            web scraping, \
#            gensim, \
#            and pipeline.'
#    elif article_topic == 'neural networks':
#        keywords = 'machine learning, \
#            artificial intelligence, \
#            neural networks,\
#            and deep learning.'
#    elif article_topic == 'regressors, classifiers, and clustering':
#        keywords = 'machine learning, \
#        scoring, \
#        and confusion matrix.'
#    else:
#        keywords = 'Err.'
#    
#    suggestion3 = "Your article appears to be about the topic of {a}. \
#        Tags that may be relevant include: {b}".format(a=article_topic,b=keywords)
#    
#    
#    return suggestion3

	
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
        
        try:
            title = soup.title.string
        except:
            title = 'nan'
        
        try:
            text_all = str(' ')
            text_blocks = soup.findAll('p')
            for i in range(0,len(text_blocks)):
                text_all = text_all + str(text_blocks[i].contents)[3:-3] + str(' ')
        except:
            text_all = 'nan'
        
        try:
            code_all = str(' ')
            code_blocks = soup.findAll('code')
            for i in range(0,len(code_blocks)):
                code_all = code_all + str(code_blocks[i].contents)[2:-2]
        except:
            code_all = 'nan'
			
        try:
            imagesCount = soup.findAll('img')
        except:
            imagesCount = float('nan')
            
        time_present = datetime.now()
        time_string = time_present.strftime("%Y-%m-%d %I:%M:%S %p")
            
        article_processed = pd.DataFrame([{'title' : title,
                                           'text' : text_all,
                                           'codeBlock' : code_all,
                                           'firstPublishedDatetime' : time_string,
                                           'postId' : 'draft_article',
										   'imagesCount' : imagesCount}],
                             columns = ["title","text","codeBlock","firstPublishedDatetime","postId","imagesCount"],
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
                
class TopicModellingSklearn:
    def __init__(self,text,min_df,max_df,n_components,random_state):
        """ 
        Initialize class. 
        
        Arguments:
            text - DF column containing text block
            min_df = Minimum number of articles the word must appear in for the
                word to be considered.
            max_df = Threshold for unique words to considered (drop words 
               appearing too frequently, as in stopwords)
            n_topics = Number of topics to consider
            random_seed = Random seed to use for the modelling
        """
        # Set up internal class variables
        self.text = text
        self.min_df = min_df
        self.max_df = max_df
        self.n_components = n_components
        self.random_state = random_state
        
        # Fit an LDA model
        self.LDA_model, self.word_frequency, self.vocabulary = self.LDA_model()

            
    def LDA_model(self):
        """ Fit text to an LDA model """
        stop_words_all = list(nltk.corpus.stopwords.words('english'))
        print(len(stop_words_all))
        stop_words_new = ["new","like","example","see","code",
                          "use","used","using","user","one","two","also",
                          "analysis","data","dataset","row","column",
                         "set","list","index","item","array",
                          "let","input","return","function","python",
                         "panda","package","number","would","figure","make","get"]
        stop_words_all.extend(stop_words_new)
        print(len(stop_words_all))
        
        word_frequency = CountVectorizer(min_df = self.min_df,
                                        stop_words=stop_words_all)
        vocabulary = word_frequency.fit_transform(
                self.text.values.astype('U'))
        
        LDA = LatentDirichletAllocation(n_components = self.n_components,
                                        random_state = self.random_state)
        LDA_model = LDA.fit(vocabulary)
        