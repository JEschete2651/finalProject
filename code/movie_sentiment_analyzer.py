
import numpy as np
import pandas as pd # Used for processing data that is in the CVS
import os
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



class MovieSetimentAnalyzer:
    def __init__(self):
        print("RD: NLP model")

    def load_kaggle_data(self):
        try:
            df = pd.read_csv("finalProject/DataSet/kaggle_dataset")
        except FileExistsError:
                print("File not found")
                exit()

