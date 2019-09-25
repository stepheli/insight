class TopicModellingSklearn:
    def __init__():
        """ 
        Initialize class. 
        
        Arguments:
            text - DF column containing text block
            min_df = Minimum number of articles the word must appear in for the
                word to be considered.
            max_df = Threshold for unique words to considered (drop words 
               appearing too frequently, as in stopwords)
            n_topics = Number of topics to consider
            random_seed = Random seed to use for the modelling
        """
        # Set up internal class variables
        self.text = text
        self.min_df = min_df
        self.max_df = max_df
        self.n_components = n_components
        self.random_state = random_state
        
        # Fit an LDA model
        self.LDA_model, self.word_frequency, self.vocabulary = self.LDA_model()

            
    def LDA_model(self):
        """ Fit text to an LDA model """
        stop_words_all = list(nltk.corpus.stopwords.words('english'))
        print(len(stop_words_all))
        stop_words_new = ["new","like","example","see","code",
                          "use","used","using","user","one","two","also",
                          "analysis","data","dataset","row","column",
                         "set","list","index","item","array",
                          "let","input","return","function","python",
                         "panda","package","number","would","figure","make","get"]
        stop_words_all.extend(stop_words_new)
        print(len(stop_words_all))
        
        word_frequency = CountVectorizer(min_df = self.min_df,
                                        stop_words=stop_words_all)
        vocabulary = word_frequency.fit_transform(
                self.text.values.astype('U'))
        
        LDA = LatentDirichletAllocation(n_components = self.n_components,
                                        random_state = self.random_state)
        LDA_model = LDA.fit(vocabulary)
        