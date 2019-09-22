# -*- coding: utf-8 -*-

# Import packages
from bs4 import BeautifulSoup
#import pandas as pd
#import numpy as np
import os
#import re

filedir_input = os.path.dirname(os.path.realpath('__file__'))
filedir_revised = os.path.join('../data/webpages')
article="Can Deep Learning Perform Better Than Pigeons_ - Towards Data Science.html"
article_path = os.path.join(filedir_revised, article)

with open(article_path,"r", encoding="utf-8") as fh:
    page = fh.read()
    fh.close()
        
soup = BeautifulSoup(page, 'html.parser')      

try:
    urlname = soup.find_all("meta",{'property':'og:url'})
    url = []
    for item in urlname:
        url.append(item["content"])
    url=url[0]
except:
    pass
