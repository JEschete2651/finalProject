import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tkinter import filedialog, Tk

nltk.download('stopwords')

class MovieSentimentAnalyzer:
    def __init__(self, lexicon: Dict[str, float]):
        self.lexicon = lexicon
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.negation_words = {"not", "no", "never", "none", "n't"}
        self.negation_scope = 3
        print("Sentiment analyzer initialized with lexicon.")

    def load_kaggle_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            print(f"Loaded dataset with {len(df)} entries.")
            return df
        except FileNotFoundError:
            print(f"File not found: {path}")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = text.split()
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def apply_negation_handling(self, tokens: List[str]) -> List[Tuple[str, bool]]:
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
        score = 0.0
        for word, negated in tokens_with_negation:
            base_score = self.lexicon.get(word, 0.0)
            if negated:
                base_score *= -1
            score += base_score
        return score

    def classify_sentiment(self, score: float) -> str:
        if score >= 0:
            return "positive"
        else:
            return "negative"

    def analyze_review(self, text: str) -> Tuple[str, float]:
        tokens = self.preprocess_text(text)
        tokens_with_neg = self.apply_negation_handling(tokens)
        score = self.compute_sentiment_score(tokens_with_neg)
        label = self.classify_sentiment(score)
        return label, score

    def run_unsup_training_gui(self):
        def get_file(title="Select file"):
            root = Tk()
            root.withdraw()
            path = filedialog.askopenfilename(title=title)
            root.update()
            return path

        print("Select labeledBow.feat file...")
        labeled_path = get_file("Select labeledBow.feat")
        print("Select unsupBow.feat file...")
        unsup_path = get_file("Select unsupBow.feat")
        print("Select imdb.vocab file...")
        vocab_path = get_file("Select imdb.vocab")

        def load_vocab(vocab_file):
            with open(vocab_file, 'r') as f:
                return [line.strip() for line in f]

        def load_feat_file(filepath, vocab_size):
            data, labels = [], []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = list(map(int, line.strip().split()))
                    label = 1 if parts[0] > 6 else 0 if parts[0] < 5 else None
                    if label is None: continue
                    features = np.zeros(vocab_size)
                    for i in range(1, len(parts), 2):
                        features[parts[i]] = parts[i+1]
                    data.append(features)
                    labels.append(label)
            return np.array(data), np.array(labels)

        def load_unsup_feat(filepath, vocab_size):
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = list(map(int, line.strip().split()))
                    features = np.zeros(vocab_size)
                    for i in range(0, len(parts), 2):
                        features[parts[i]] = parts[i+1]
                    data.append(features)
            return np.array(data)

        vocab = load_vocab(vocab_path)
        vocab_size = len(vocab)
        X_labeled, y_labeled = load_feat_file(labeled_path, vocab_size)
        X_unsup = load_unsup_feat(unsup_path, vocab_size)

        print("Fitting PCA on unsupervised data...")
        pca = PCA(n_components=100)
        pca.fit(X_unsup)

        print("Transforming labeled data...")
        X_reduced = pca.transform(X_labeled)

        print("Splitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_labeled, test_size=0.2, random_state=42)

        print("Training logistic regression...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        print("Evaluating on test set...")
        predictions = clf.predict(X_test)
        print(classification_report(y_test, predictions))


def main():
    lexicon = {}  # TODO: Load from file or define
    analyzer = MovieSentimentAnalyzer(lexicon)
    analyzer.run_unsup_training_gui()

if __name__ == '__main__':
    main()
