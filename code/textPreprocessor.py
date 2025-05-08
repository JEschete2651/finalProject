
# This class is responsible for preprocessing the data to handle the following 

# // {
# 1.Text cleaning 
# 2. Stopword removal 
# 3. Lemmatization and negation handling
#  // }

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english")) - {"not", "no"}
        self.lemmatizer = WordNetLemmatizer()
        self.negation_words = {
            "not", "no", "never", "none", "n't", "without",
            "neither", "nor", "hardly", "barely", "scarcely",
            "don't", "doesn't", "isn't", "wasn't", "weren't",
            "can't", "won't", "shouldn't", "couldn't", "wouldn't"
        }
        self.negation_scope = 3
        self.pca = None
        self.clf = None
        self.settings = Settings()
    
    def preprocess(self, text: str):
        if settings.key_exist("log_preprocess") is True:
            print(f"Original text: {text}")
            
        # Lower case the text to process
        lower_cased_text = text.lower()
        print(f'Lowercased text: {lower_cased_text}')
        
        # Punctuation marks to remove from the lower cased text.
        punctuation = string.punctuation.replace("'", "")
        
        # This is the cleaned text without the punctuation marks.
        cleaned_text = re.sub(f"[{re.escape(punctuation)}]", "", lower_cased_text)
        
        # Print out the punctuations removed
        if settings.key_exist("log_preprocess") is True:
            print(f"Punctuation removed: {cleaned}")
            
        # Create tokens
        tokens = cleaned_text.split()
        if settings.key_exist("log_preprocess") is True:
            print(f"Tokenized: {tokens}")
        
        # Lemmatize 
        lemmatized = [
            for w in tokens:
                if (w in self.negation_words) or (w not in self.stop_words):
                    self.lemmatizer.lemmatize(w)
        ]
        if settings.get("log_preprocess", True):
           print(f"Lemmatized tokens: {lemmatized}")
        
        return lemmatized