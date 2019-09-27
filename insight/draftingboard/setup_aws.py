from flask import Flask, render_template, request
from draftingboard import process_text_aws
import pickle
#from draftingboard import TopicModellingSklearn
#from draftingboard import recommender

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
           user_input = '<title> DraftingBoard: A tool for optimizing blog posts </title> \
           <p> Writing clean code is hard, but writing clean text about clean code can be \
           even more challenging. This tool is designed to take a look at the content of the \
           first draft of a blog post which contains both text and code and offer suggestions \
           about what to be improved to make it more accessible. For both authors and website \
           hosts, this offers benefits in terms of higher readership, engagement, and \
           advertising revenue.</p> <p> Here\'s a sample code snippet for analysis. \ <code> \
           import pandas as pd for line in self.lines: # Reset tracking variables line_code = 0 \
           line_comment = 0 # If the comment character is not present, assume line of code \
           if line.find(commentchar) == -1: line_code = 1 # If a comment character is present, \
           determine where it falls to see if line is a pure comment, or mixed code + comment \
           else: comment_start = line.index(commentchar) if comment_start == 0: line_code = 0 \
           line_comment = 1 </code> <img src = "testing.jpg">'
           
           analysed_text = process_text_aws.master_function(user_input)
           similar_articles = analysed_text[0][0]
           suggestions = analysed_text[1]
           return render_template("index_v2.html",
                              my_input=user_input,
                              my_output=user_input,
                              article1_title=str(similar_articles.iloc[0]),
							  article2_title=str(similar_articles.iloc[1]),
							  article3_title=str(similar_articles.iloc[2]),
                              sugg1=str(suggestions[0]),
                              sugg2=str(suggestions[1]),
                              sugg3=str(suggestions[2]),
                              results_plot='/static/img/output.png',
                              my_form_result="NotEmpty")
           
       else:           
           analysed_text = process_text_aws.master_function(user_input)
           similar_articles = analysed_text[0][0]
           suggestions = analysed_text[1]
           return render_template("index_v2.html",
                              my_input=user_input,
                              my_output=user_input,
                              article1_title=str(similar_articles.iloc[0]),
							  article2_title=str(similar_articles.iloc[1]),
							  article3_title=str(similar_articles.iloc[2]),
                              sugg1=str(suggestions[0]),
                              sugg2=str(suggestions[1]),
                              sugg3=str(suggestions[2]),
                              results_plot='/static/img/output.png',
                              my_form_result="NotEmpty")

# start the server with the 'run()' method
if __name__ == "__main__":   
    app.run(debug=True) #will run locally http://127.0.0.1:5000/     