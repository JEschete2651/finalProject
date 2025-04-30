import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Tuple, Dict

# Make sure necessary NLTK data is downloaded
nltk.download('stopwords')

class MovieSentimentAnalyzer:
    def __init__(self, lexicon: Dict[str, float]):
        self.lexicon = lexicon
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.negation_words = {"not", "no", "never", "none", "n't"}
        self.negation_scope = 3  # number of words after negation affected
        print("Sentiment analyzer initialized with lexicon.")

    def load_kaggle_data(self, path: str) -> pd.DataFrame:
        """
        Load dataset from given path.
        """
        try:
            df = pd.read_csv(path)
            print(f"Loaded dataset with {len(df)} entries.")
            return df
        except FileNotFoundError:
            print(f"File not found: {path}")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocessing: lowercase, punctuation removal, tokenization, stop word removal, stemming.
        """
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = text.split()
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def apply_negation_handling(self, tokens: List[str]) -> List[Tuple[str, bool]]:
        """
        Tag tokens with negation context.
        """
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in self.negation_words:
                for j in range(i + 1, min(i + 1 + self.negation_scope, len(tokens))):
                    result.append((tokens[j], True))
                i += self.negation_scope
            else:
                result.append((token, False))
                i += 1
        return result

    def compute_sentiment_score(self, tokens_with_negation: List[Tuple[str, bool]]) -> float:
        """
        Aggregate sentiment score with negation consideration.
        """
        score = 0.0
        for word, negated in tokens_with_negation:
            base_score = self.lexicon.get(word, 0.0)
            if negated:
                base_score *= -1
            score += base_score
        return score

    def classify_sentiment(self, score: float) -> str:
        """
        Convert sentiment score to label.
        """
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    def analyze_review(self, text: str) -> Tuple[str, float]:
        """
        Analyze a single review and return label and score.
        """
        tokens = self.preprocess_text(text)
        tokens_with_neg = self.apply_negation_handling(tokens)
        score = self.compute_sentiment_score(tokens_with_neg)
        label = self.classify_sentiment(score)
        return label, score
