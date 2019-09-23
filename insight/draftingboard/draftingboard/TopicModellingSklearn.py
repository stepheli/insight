class TopicModellingSklearn:
    def __init__(self,text,text_proc_check,min_df,max_df,n_components,random_state):
        """ 
        Initialize class. 
        
        Arguments:
            text - DF column containing text block
            text_proc_check = Boolean controlling whether text is preprocessed
            min_df = Minimum number of articles the word must appear in for the
                word to be considered.
            max_df = Threshold for unique words to considered (drop words 
               appearing too frequently, as in stopwords)
            n_topics = Number of topics to consider
            random_seed = Random seed to use for the modelling
        """
        # Set up internal class variables
        self.text = text
        self.text_proc_check = text_proc_check
        self.min_df = min_df
        self.max_df = max_df
        self.n_components = n_components
        self.random_state = random_state
        
        # Pre-process text if requested
        if self.text_proc_check == True:
            self.text_preprocess()
            
            
        # Fit an LDA model
        self.LDA_model, self.word_frequency, self.vocabulary = self.LDA_model()
    
    def text_preprocess(self):
        """ 
        Arguments:
            text - pandas series where each entry is the unprocessed text of
                a given article.
            
        Outputs:
            text_proc - pandas series where the article text has been processed
                based on the following workflow:
                    (1) Case consistency (all lowercase)
                    (2) Lemmatizing
        """   
        lemmatizer = WordNetLemmatizer()
        
        for article in range(0,len(self.text)):
            # Print a status update
            if np.remainder(article,100) == 0:
                print("Pre-processing requested. Lemmatizing checkpoint {}.".format(article))
#                print("Original article: {}".format(self.text[article][0:500]))
            
            # Process article text, overwrite original text
            article_text_proc = []
            article_text = self.text[article].split(" ")
            for word in article_text:
                word = word.lower()
                article_text_proc.append(lemmatizer.lemmatize(word))
            self.text[article] = " ".join(article_text_proc)
            
#            if np.remainder(article,100) == 0:
#                print("Lemmatized article: {}".format(self.text[article][0:500]))
            
    def LDA_model(self):
        """ Fit text to an LDA model """
        word_frequency = CountVectorizer(min_df = self.min_df,
                                         stop_words = 'english')
        vocabulary = word_frequency.fit_transform(
                self.text.values.astype('U'))
        
        LDA = LatentDirichletAllocation(n_components = self.n_components,
                                        random_state = self.random_state)
        LDA_model = LDA.fit(vocabulary)
        
        return LDA_model, word_frequency, vocabulary
