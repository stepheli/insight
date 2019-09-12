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
def filterArticles(articles, nPublications, pTop):
    """ 
    From a large DataFrame of posts from Medium-affiliated publications, 
    extract those from the most popular articles and publications.
    
    Arguments:
        articles = DataFrame containing post info
        nPublications = Number of publications with the most number of total 
            posts to filter based on
        pTop = Percentage threshold for most popular articles (evaluated based 
            on number of total claps) 
    """
    # Remove articles without an associated publication name. 
    articles = articles[articles.publicationname.notnull()]
    
    # Construct list of unique publications, sort according to most prolific
    pubCount = articles.groupby(["publicationname"]).size().reset_index(name="counts")
    pubCount = pubCount.sort_values(["counts"],ascending=False)
    
    print("Initial input: {} articles across {} publications".format(
            len(articles),len(pubCount)))
    
    # Filter articles to get only those from the top publications
    topPubs = pubCount["publicationname"][0:nPublications].values.tolist()
    articlesFiltered = articles[articles["publicationname"].isin(topPubs)]
    
    # Remove duplicates identified based on identical post ID
    articlesFiltered = articlesFiltered.drop_duplicates(subset="postId", keep='first')
    
    # Filter articles to get only those that are most popular
    nArticles = int(np.ceil(pTop*len(articlesFiltered)))
    articlesFiltered = articlesFiltered.sort_values(["totalClapCount"],ascending=False)
    articlesFiltered = articlesFiltered.iloc[0:nArticles,:]
    
    # Pull out only the columns to be used in further analysis
    articlesFiltered = articlesFiltered[["firstPublishedDatetime",
                   "imageCount",
                   "linksCount",
                   "postId",
                  "recommends",
                  "responsesCreatedCount",
                  "tagsCount",
                  "text",
                  "title",
                  "totalClapCount",
                  "wordCount",
                  "publicationname"]]
    
    print(" ")
    print("Filter settings: Most popular {}% {} most prolific publications".format(
            pTop*100, nPublications))
    print("{} articles remaining".format(len(articlesFiltered)))    
    
    return articlesFiltered

class ArticleTextAnalyser:
    """
    Process the text of an article to extract some secondary statistics 
    (word length, sentence length, unique word count, etc.)
    """
    def __init__(self,articleText):
        """ 
        Initialize class. Receive article text as input, split it up into 
        words and sentences for subfunctions to process further.
        
        Arguments:
            self - Internal class references
            articletext - Single string variable containing all the article text. 
            Does not have to be preprocessed to remove stray formatting 
            characters.
        """
        stopset = set(nltk.stopwords.words('english'))
        
        self.articleText = articleText
        self.sentences = articleText.split(".")
        self.words = nltk.word_tokenize(articleText)
        self.words = [word for word in self.words if not word in stopset]
        
        
    
    def analyseSentences(self):
        pass
    
    def analyseWords(self):
        pass
        

# Import data
fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join('../data/raw/Medium_AggregatedData.csv')
filename = os.path.abspath(os.path.realpath(filename))
articles = pd.read_csv(filename)

# Filter articles
articlesFiltered = filterArticles(articles,5,0.1)
    
# For the filtered posts, construct a histogram of the selected characteristics
fig = plt.figure()
scatter_matrix(articlesFiltered)
plt.show()

## Analyse text
stopset = set(stopwords.words('english'))

articleText = articlesFiltered["text"].iloc[0]
sentences = articleText.split(".")
words = nltk.word_tokenize(articleText)
print(len(words))
#words = [word for word in words if not word in stopset]
#print(len(words))

        
