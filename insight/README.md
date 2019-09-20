# DraftingTable

# Overview 
Writing clean code is hard work. Writing clear text about clean code is equally challenging. The goal of this project is to simplify this process by analysing a prospective blog post against the analytical metrics of the top 10% of most popular blog posts (determined by number of claps) for the top 5 most prolific blogs under the Medium umbrella:
1. Towards Data Science
2. Hacker Noon
3. Becoming Human: Artificial Intelligence Magazine
4. Chatbots Life
5. Chatbots Magazine

# Data Sources & Processing

The data set used in this project is compiled from three sources:

1) A Kaggle dataset of 279,577 articles of scraped posts put together by  Aiswarya Ramachandran which contains scraped posts from Medium-affiliated plots tagged with AI, Machine Learning, Datascience, or Artificial Intelligence from September 2017 - September 2018.
(https://www.kaggle.com/aiswaryaramachandran/medium-articles-with-content/downloads/medium-articles-with-content.zip/2)

This data set was processed to perform exploratory data analysis and filter the data set to retain only article with an associated publication name and at least one code block. This code can be found in the script:
> scripts/01-article-filterer.py
This script outputs a truncated CSV containing the processed data to:
> data/processed/filtereddata.csv

The filtered data was processed further to perform exploratory NLP and topic modelling analysis. This was done in two scripts:
> scripts/02-filtered-article-analyser.py
>> This script calculates analytical metrics for the text(average sentence length, word count, etc.) and code blocks (estimated language, ratio of code/comment lines, average comment length). These are outputted to the files:
>>> data/processed/articles_javascript.csv
>>> data/processed/articles_python.csv
>>>> (Pickled in draftingboard/articles_python.pickle)
>>> data/processed/articles_sql.csv

2) A collection of 200+ more recent articles scraped specifically for this project from the publications represented in the above data set (Towards Data Science, Hacker Noon, Insight Data Science, etc.)

This scraping was performed and the relevant data blocks extracted in the script:
> scripts/07-web-scraping-recent.py
This script outputs a truncated CSV containining the processed data to:
> data/processed/articles_scraped.csv

3) The content of a draft post collected from the end-user through a Flask app, which is compared to the historical data set. The backend of the Flask app can be found in:
> draftingboard/

# Methodology & Algorithms
What features of code and content translate to clear, accessible content? Do these features directly translate to success? To answer these questions, several natural language processing methods were applied:

(1) Topic modelling on articles sharing the same coding language. Given the imbalance in articles across coding languages (Python >> Javascript > SQL), only the Python articles were selected for this analysis. The method of Latent Dirichlet Allocation (LDA) as implemented by the sklearn analysis was selected for this approach, and was implemented in:
> scripts/03-topic-modelling-python.py

The initial model (model_alpha) considered text with stopwords filtered out but no additional pre-processing, and identified topics which correlated well with clear subfields represented in the data science articles: natural language processing, neural networks, supervised learning,  and general data science. 

The initial refinement (model_beta) considered additional pre-processing steps of stemming/lemmatizing, but displayed more overlap between the top keywords associated with each topic and less distinction between the topics.

model_alpha was used to assign both the historical articles and draft posts to a given topic.