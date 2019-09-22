# -*- coding: utf-8 -*-

# Import packages
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re


class ScrapeArticle:
    """ 
    Purpose: For a locally saved HTML article, scrape the contents to
    extract title, text, figure count, etc. Output this as a dataframe.
    """
    def __init__(self,filename):
        self.filename = filename
        
        self.article_scraped = self.initial_parse()
                
    def initial_parse(self):
        """
        Purpose: Open the file, extract metrics using BeautifulSoup, clean 
        extracted quantities up into a DF for later use.
        
        Inputs:
            - self.filename = HTML filename to be scraped. Includes full filepath.
            
        Outputs:
            - article_scraped = pandas DF with extracted article metrics.
        """
        with open(self.filename,"r", encoding="utf-8") as fh:
            page = fh.read()
            fh.close()
            
        soup = BeautifulSoup(page, 'html.parser')
        
        title_text = []
        title = soup.find_all("title")
        for item in title:
            title_text.append(item.get_text())
        
        content_text = []
        content = soup.find_all("p")
        for item in content:
            content_text.append(item.get_text())
        
        content_text_merged = str(" ")
        for i in range(0,len(content_text)):
            content_text_merged = content_text_merged + content_text[i] + str(" ")
        
        figure = soup.find_all("figure")
        figure_count = len(figure)
                
        # The below code assumes that the article metadata is embedded in an 
        # HTML tag with the format below
        # <script type="application/ld+json" data-rh="true">
        # Try/Except logic is set up to catch article which do not behave this
        # way, in which case the default values of nan will not be overwritten.
        datePublished = 'nan'
        postId = 'nan'
        publicationname = 'nan'
        url = 'nan'

        try:
            metadata = soup.find_all("script", 
                         {'data-rh' : 'true'})
            metadata_split = str(metadata)
            metadata_split = re.split(': |,',metadata_split)
            metadata_split = [attribute.split(":") for attribute in metadata_split]
         
            
            for i in range(0,len(metadata_split)):
                if metadata_split[i][0].find("datePublished") > 0:
                    datePublished = metadata_split[i][1]
                    datePublished = datePublished[1:-1]
                elif metadata_split[i][0].find("identifier") > 0:
                    postId = metadata_split[i][1]
                    postId = postId[1:-1]
                elif metadata_split[i][0].find("Publication") > 0:
                    publicationname = metadata_split[i][1]
                    publicationname = publicationname.split("-")
                    publicationname = " ".join(publicationname)
                    publicationname = publicationname.title()[0:-1]
        except:
            pass

     
        article_scraped = pd.DataFrame({'title' : title_text,
                                        'text' : content_text_merged,
                                        'figureCount' : figure_count,
                                        'publicationname' : publicationname,
                                        'postId' : postId,
                                        'firstPublishedDatetime' : datePublished},
                                        columns = ["title",
                                          "text",
                                          "figureCount",
                                          "publicationname",
                                          "postId",
                                          "firstPublishedDatetime"])
    
        return article_scraped
             

filedir_input = os.path.dirname(os.path.realpath('__file__'))
filedir_revised = os.path.join('../data/webpages')

article_list = []
for article in os.listdir(filedir_revised):
    if article.endswith(".html"):
        article_list.append(os.path.join(filedir_revised, article))
      
articles_scraped = pd.DataFrame(columns = ['title',
                                           'text',
                                           'figureCount',
                                           'publicationname',
                                           'postId',
                                           'firstPublishedDatetime',
                                           'url'])
articleno = 0
for article in article_list:
    articleno = articleno + 1
    try:
        if np.remainder(articleno,50) == 0:
            print("Scraping checkpoint: processing article {} of {}".format(articleno, len(article_list)))
        article_scraped = ScrapeArticle(article).article_scraped
        articles_scraped = articles_scraped.append(article_scraped)
    except:
        print("Error scraping article: {}".format(article))

# Export scraped articles to new csv for faster import in later scripts
filedir_output = os.path.dirname(os.path.realpath('__file__'))
filename_output = os.path.join('../data/processed/articles_scraped.csv')
filename_output = os.path.abspath(os.path.realpath(filename_output))
articles_scraped.to_csv(path_or_buf=filename_output)

