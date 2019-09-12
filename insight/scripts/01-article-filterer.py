# Import packages
import pandas as pd 
#from pandas.plotting import scatter_matrix
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)


# Define functions
def filter_articles(articles, n_publications, p_top):
    """ 
    From a large DataFrame of posts from Medium-affiliated publications, 
    extract those from the most popular articles and publications.
    
    Arguments:
        articles = DataFrame containing post info
        nPublications = Number of publications with the most number of total 
            posts to filter based on
        pTop = Percentage threshold for most popular articles (evaluated based 
            on number of total claps) 
        
    Outputs:
        articles_filtered = DataFrame of filtered articles, keeping only the 
            columns to be used in further analysis.
    """
    # Remove articles without an associated publication name. 
    articles = articles[articles.publicationname.notnull()]
    
    # Construct list of unique publications, sort according to most prolific
    pub_count = articles.groupby(["publicationname"]).size().reset_index(
            name="counts")
    pub_count = pub_count.sort_values(["counts"],ascending=False)
    
    print("Initial input: {} articles across {} publications".format(
            len(articles),len(pub_count)))
    
    # Filter articles to get only those from the top publications
    top_pubs = pub_count["publicationname"][0:n_publications+1].values.tolist()
    articles_filtered = articles[articles["publicationname"].isin(top_pubs)]
    
    # Remove duplicates identified based on identical post ID
    articles_filtered = articles_filtered.drop_duplicates(
            subset="postId", keep='first')
    
    # Filter articles to get only those that are most popular
    n_articles = int(np.ceil(p_top*len(articles_filtered)))
    articles_filtered = articles_filtered.sort_values(["totalClapCount"],
                                                    ascending=False)
    articles_filtered = articles_filtered.iloc[0:n_articles,:]
    
    # Pull out only the columns to be used in further analysis
    articles_filtered = articles_filtered[["firstPublishedDatetime",
                   "imageCount",
                   "linksCount",
                  "tagsCount",
                  "text",
                  "title",
                  "wordCount",
                  "publicationname",
                  "totalClapCount",
                  "recommends",
                  "responsesCreatedCount"]]
    
    print(" ")
    print("Filter settings: Most popular {}% {} most prolific publications".format(
            p_top*100, n_publications))
    print("{} articles remaining".format(len(articles_filtered)))    
    
    return articles_filtered

# Import data
filedir_input = os.path.dirname(os.path.realpath('__file__'))
filename_input = os.path.join('../data/raw/Medium_AggregatedData.csv')
filename_input = os.path.abspath(os.path.realpath(filename_input))
articles = pd.read_csv(filename_input)

# Filter articles
articles_filtered = filter_articles(articles,5,0.1)

# Export filtered articles to new csv for faster import in later scripts
filedir_output = os.path.dirname(os.path.realpath('__file__'))
filename_output = os.path.join('../data/processed/filtereddata.csv')
filename_output = os.path.abspath(os.path.realpath(filename_output))
articles_filtered.to_csv(path_or_buf=filename_output)

# For initial visualization, pull numerical data for scatter matrix
articles_selected = articles_filtered[["imageCount",
                                       "tagsCount",
                                       "wordCount",
                                       "totalClapCount"]]
fig = plt.figure()
sns.pairplot(articles_selected)
plt.show()

