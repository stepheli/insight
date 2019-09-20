from flask import Flask, render_template, request
#from bs4 import BeautifulSoup
#import os
#import numpy as np
#import pandas as pd
#from datetime import datetime
#import pickle
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import WhitespaceTokenizer
#from sklearn.feature_extraction.text import CountVectorizer
#from joblib import load
#import string
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import seaborn as sns
#sns.set(style="ticks", color_codes=True, font_scale=0.95)

from draftingboard import process_text

# Create the application object
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output',methods=["GET","POST"])
def tag_output():
#       
       # Pull input
       user_input =request.form["user_input"]        
       
       # Case if empty
       if user_input == '':
           return render_template("index.html",
                                  my_input = user_input,
                                  my_form_result="Empty")
       else:
           some_number=3
           processed_text = process_text.master_function(user_input)
           return render_template("index.html",
                              my_input=user_input,
                              my_output=user_input,
                              my_number=some_number,
                              results_plot='/static/img/output.png',
                              my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/

