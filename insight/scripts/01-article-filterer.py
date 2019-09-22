# Import packages
import pandas as pd 
#from pandas.plotting import scatter_matrix
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1)


def filter_articles_publications(articles, n_publications, p_top):
    """ 
    From a large DataFrame of posts from Medium-affiliated publications, 
    extract those from the most prolific publications.
    
    Arguments:
        articles = DataFrame containing post info
        n_publications = Number of publications with the most number of total 
            posts to filter based on
        p_top = Percentage threshold for most popular articles (evaluated based 
            on number of total claps) 
        titlekeyword = Word to look for in the title
        
    Outputs:
        articles_filtered = DataFrame of filtered articles, keeping only the 
            columns to be used in further analysis.
    """
    # Remove articles without an associated publication name, select only those 
    # with data explicitly in the name
    articles = articles[articles.publicationname.notnull()]
    articles = articles[articles["publicationname"].str.contains("Data")]
    
    # Construct list of unique publications, sort according to most prolific
    pub_count = articles.groupby(["publicationname"]).size().reset_index(
            name="counts")
    pub_count = pub_count.sort_values(["counts"],ascending=False)
    print(pub_count.head())
    
    print("Initial input: {} articles across {} publications".format(
            len(articles),len(pub_count)))
    
    # Filter articles to get only those from the top publications
    top_pubs = pub_count["publicationname"][0:n_publications+1].values.tolist()
    print(top_pubs)
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
    articles_filtered = articles_filtered[["postId",
                  "firstPublishedDatetime",
                  "imageCount",
                  "linksCount",
                  "tagsCount",
                  "text",
                  "title",
                  "wordCount",
                  "codeBlock",
                  "codeBlockCount",
                  "publicationname",
                  "totalClapCount",
                  "recommends",
                  "responsesCreatedCount"]]
    
    print(" ")
    print("Filter settings: Most popular {}% {} most prolific publications".format(
            p_top*100, n_publications))
    print("{} articles remaining".format(len(articles_filtered)))    
    
    return articles_filtered

def filter_articles_codeblocks(articles, n_codeblocks, p_top):
    """ 
    From a large DataFrame of posts from Medium-affiliated publications, 
    extract the most popular posts which also contain code.
    
    Arguments:
        articles = DataFrame containing post info
        n_codeblocks = Minimum number of codeblocks in post 
        p_top = Percentage threshold for most popular articles (evaluated based 
            on number of total claps) 
        titlekeyword = Word to look for in the title
        
    Outputs:
        articles_filtered = DataFrame of filtered articles, keeping only the 
            columns to be used in further analysis.
    """
    # Preliminary cleaning: remove articles with a missing publication name or
    # not in English
    articles = articles[articles.publicationname.notnull()]
    articles = articles[articles["language"] == "en"]
    
    print("Initial input: {} articles".format(len(articles)))
    
    # Remove articles without sufficient codeblocks
    articles_filtered = articles[articles["codeBlockCount"] >= n_codeblocks]
            
    # Filter articles to get only those that are most popular
    articles_filtered = articles_filtered.drop_duplicates(subset="postId", keep='first')
    n_articles = int(np.ceil(p_top*len(articles_filtered)))
    articles_filtered = articles_filtered.sort_values(["totalClapCount"],
                                                    ascending=False)
    articles_filtered = articles_filtered.iloc[0:n_articles,:]
    
    # Pull out only the columns to be used in further analysis
    articles_filtered = articles_filtered[["postId",
                  "firstPublishedDatetime",
                  "imageCount",
                  "linksCount",
                  "tagsCount",
                  "text",
                  "title",
                  "wordCount",
                  "codeBlock",
                  "codeBlockCount",
                  "publicationname",
                  "totalClapCount",
                  "recommends",
                  "url",
                  "responsesCreatedCount"]]
    
    print(" ")
    print("Filter settings: Most popular {}% with minimum {} codeblock".format(
            p_top*100, n_codeblocks))
    print("{} articles remaining".format(len(articles_filtered)))    
    
    return articles_filtered
    
# Import data
filedir_input = os.path.dirname(os.path.realpath('__file__'))
filename_input = os.path.join('../data/raw/Medium_AggregatedData.csv')
filename_input = os.path.abspath(os.path.realpath(filename_input))
articles = pd.read_csv(filename_input)
articles["firstPublishedDatetime"] = pd.to_datetime(
        articles["firstPublishedDatetime"])

# Filter articles
articles_filtered = filter_articles_codeblocks(articles,1,1)

# Duplicate the column containing a unique identifier (postId) so it can be 
# called directly later, then set it as the DF index
articles_filtered["postId2"] = articles_filtered["postId"]
articles_filtered.set_index("postId",inplace=True)

# Convert datetimeobjects to hour of post for later plotting
articles_timelist = []
articles_filtered_timelist = []
for i in range(0,len(articles)):
    articles_timelist.append(
            articles["firstPublishedDatetime"].iloc[i].time().hour)
for i in range(0,len(articles_filtered)):
    articles_filtered_timelist.append(
            articles_filtered["firstPublishedDatetime"].iloc[i].time().hour)

# Export filtered articles to new csv for faster import in later scripts
filedir_output = os.path.dirname(os.path.realpath('__file__'))
filename_output = os.path.join('../data/processed/filtereddata.csv')
filename_output = os.path.abspath(os.path.realpath(filename_output))
articles_filtered.to_csv(path_or_buf=filename_output)

# Initial visualization: Difference in histograms of post characteristics 
# between all and popular posts
fig = plt.figure(figsize=(18,7))

ax1a = fig.add_subplot(2,4,1)
sns.distplot(articles_timelist,bins=24,kde=False)
plt.xticks(np.arange(0, 25, 6)) 
ax1a.set_ylabel('All Posts')

ax1b = fig.add_subplot(2,4,5)
sns.distplot(articles_filtered_timelist,bins=24,kde=False)
ax1b.set_xlabel('Hour of Post')
plt.xticks(np.arange(0, 25, 6)) 
ax1b.set_ylabel('Popular Posts')

ax2a = fig.add_subplot(2,4,2)
sns.distplot(articles["tagsCount"],kde=False)
ax2a.set_xlabel('')
ax2a.set_xlim(0,5)

ax2b = fig.add_subplot(2,4,6)
sns.distplot(articles_filtered["tagsCount"],kde=False)
ax2b.set_xlabel('Tags Added')
ax2b.set_xlim(0,5)

ax3a = fig.add_subplot(2,4,3)
sns.distplot(articles["codeBlockCount"],bins=100,kde=False)
ax3a.set_xlabel('')

ax3b = fig.add_subplot(2,4,7)
sns.distplot(articles_filtered["codeBlockCount"],bins=25,kde=False)
ax3b.set_xlabel('Code Blocks')

ax4a = fig.add_subplot(2,4,4)
sns.distplot(articles["wordCount"],bins=200,kde=False)
ax4a.set_xlabel('')
ax4a.set_xlim(0,5000)

ax4b = fig.add_subplot(2,4,8)
sns.distplot(articles_filtered["wordCount"],bins=50,kde=False)
ax4b.set_xlabel('Word Count')
ax4b.set_xlim(0,5000)

plt.subplots_adjust(wspace=0.4)

# Secondary visualization: correlation between post characteristics and 
# popularity for filtered articles
articles_selected = articles_filtered[["codeBlockCount",
                                       "wordCount",
                                       "totalClapCount"]]
sns.pairplot(articles_selected,height=2,aspect=1.5)
plt.show()
