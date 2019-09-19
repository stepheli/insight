# Import packages
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend(index, method):
    id = indices[index]
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    
    articles_index = [i[0] for i in similarity_scores]
    
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return articles_all['title'].iloc[articles_index]

# Import and compile relevant data sets
# File1: articles_scraped (output of 07-web-scraping-recent.py)
# File2: articles_python (output of 02-filtered-article-analyser.py)
filedir_input = os.path.dirname(os.path.realpath('__file__'))

filename_input1 = os.path.join('../data/processed/articles_scraped.csv')
filename_input1 = os.path.abspath(os.path.realpath(filename_input1))
articles_all = pd.read_csv(filename_input1, index_col = 'postId')

filename_input2 = os.path.join('../data/processed/articles_python.csv')
filename_input2 = os.path.abspath(os.path.realpath(filename_input2))
articles_all = articles_all.append(pd.read_csv(filename_input2, index_col = 'postId'))

# Process text column to construct tfidf_matrix as input for cosine similarity
topic_frequency = TfidfVectorizer(analyzer = 'word',
                                 min_df = 1,
                                 stop_words = 'english')
tfidf_matrix = topic_frequency.fit_transform(articles_all["text"])

cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix) 

articles_all = articles_all.reset_index(drop=True)
indices = pd.Series(articles_all["text"].index)

article_test = 3
print("Original article: {}".format(articles_all["title"].iloc[article_test]))
print(" ")
print(recommend(article_test, cosine_similarity))