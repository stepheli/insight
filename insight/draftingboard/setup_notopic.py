from flask import Flask, render_template, request
from draftingboard import process_text_notopic
from draftingboard.TopicModellingSklearn import TopicModellingSklearn
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
           return render_template("index.html",
                                  my_input = user_input,
                                  my_form_result="Empty")
       else:           
           analysed_text = process_text_notopic.master_function(user_input)
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