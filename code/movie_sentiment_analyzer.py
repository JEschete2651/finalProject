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
    """
    A class to perform sentiment analysis on movie reviews using a lexicon-based approach
    and optionally train a supervised model using features derived from unsupervised data.
    """
    def __init__(self, lexicon: Dict[str, float]):
        """
        Initializes the sentiment analyzer.

        Args:
            lexicon (Dict[str, float]): A dictionary mapping words (stems) to sentiment scores.
                                         Positive scores indicate positive sentiment, negative scores
                                         indicate negative sentiment.
        """
        self.lexicon = lexicon  # Store the provided sentiment lexicon
        self.stop_words = set(stopwords.words('english'))  # Load English stop words
        self.stemmer = PorterStemmer()  # Initialize the Porter Stemmer
        # Define words that indicate negation
        self.negation_words = {"not", "no", "never", "none", "n't"}
        # Define the scope (number of words) affected by a negation word
        self.negation_scope = 3
        print("Sentiment analyzer initialized with lexicon.")

    def load_kaggle_data(self, path: str) -> pd.DataFrame:
        """
        Loads a dataset from a CSV file using pandas.

        Args:
            path (str): The file path to the CSV dataset.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded data.
                          Returns an empty DataFrame if the file is not found.
        """
        try:
            # Attempt to read the CSV file into a DataFrame
            df = pd.read_csv(path)
            print(f"Loaded dataset with {len(df)} entries.")
            return df
        except FileNotFoundError:
            # Handle the case where the file does not exist
            print(f"File not found: {path}")
            return pd.DataFrame() # Return an empty DataFrame

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesses raw text by converting to lowercase, removing punctuation,
        tokenizing, removing stop words, and stemming.

        Args:
            text (str): The raw text string to preprocess.

        Returns:
            List[str]: A list of processed (stemmed, non-stop word) tokens.
        """
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation using regex substitution
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        # Split the text into individual words (tokens)
        tokens = text.split()
        # Filter out stop words and apply stemming to the remaining words
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def apply_negation_handling(self, tokens: List[str]) -> List[Tuple[str, bool]]:
        """
        Applies simple negation handling to a list of tokens.
        Marks tokens following a negation word as negated.

        Args:
            tokens (List[str]): A list of preprocessed tokens.

        Returns:
            List[Tuple[str, bool]]: A list of tuples, where each tuple contains
                                     a token and a boolean indicating if it's negated.
        """
        result = [] # List to store (token, is_negated) tuples
        i = 0 # Index for iterating through tokens
        while i < len(tokens):
            token = tokens[i]
            # Check if the current token is a negation word
            if token in self.negation_words:
                # Mark the next 'negation_scope' tokens as negated
                for j in range(i + 1, min(i + 1 + self.negation_scope, len(tokens))):
                    # Append the subsequent token with negated flag set to True
                    result.append((tokens[j], True))
                # Skip the negation word and the words within its scope
                i += self.negation_scope # Note: This skips the negation word itself AND the scope.
                                         # Consider if the negation word itself should be added (likely not needed for scoring).
                                         # Also, check if `i += self.negation_scope + 1` might be intended if the scope starts *after* the neg word.
                                         # Current logic correctly processes words *after* negation up to scope limit.
            else:
                # If not a negation word, append the token with negated flag as False
                result.append((token, False))
                # Move to the next token
                i += 1
        return result

    def compute_sentiment_score(self, tokens_with_negation: List[Tuple[str, bool]]) -> float:
        """
        Computes the overall sentiment score for a list of tokens,
        considering negation.

        Args:
            tokens_with_negation (List[Tuple[str, bool]]): A list of (token, is_negated) tuples.

        Returns:
            float: The calculated sentiment score. Positive values indicate positive sentiment,
                   negative values indicate negative sentiment.
        """
        score = 0.0 # Initialize the total score
        # Iterate through each token and its negation status
        for word, negated in tokens_with_negation:
            # Get the base score from the lexicon, defaulting to 0.0 if word not found
            base_score = self.lexicon.get(word, 0.0)
            # If the word is marked as negated, invert its score
            if negated:
                base_score *= -1
            # Add the possibly modified score to the total
            score += base_score
        return score

    def classify_sentiment(self, score: float) -> str:
        """
        Classifies sentiment as 'positive' or 'negative' based on the score.

        Args:
            score (float): The sentiment score.

        Returns:
            str: 'positive' if score is >= 0, 'negative' otherwise.
        """
        # Classify based on a simple threshold (0)
        if score >= 0:
            return "positive"
        else:
            return "negative"

    def analyze_review(self, text: str) -> Tuple[str, float]:
        """
        Performs end-to-end sentiment analysis on a single movie review text.

        Args:
            text (str): The movie review text.

        Returns:
            Tuple[str, float]: A tuple containing the predicted sentiment label ('positive'/'negative')
                               and the calculated sentiment score.
        """
        # 1. Preprocess the text
        tokens = self.preprocess_text(text)
        # 2. Apply negation handling
        tokens_with_neg = self.apply_negation_handling(tokens)
        # 3. Compute the sentiment score
        score = self.compute_sentiment_score(tokens_with_neg)
        # 4. Classify the sentiment based on the score
        label = self.classify_sentiment(score)
        return label, score

    def run_unsup_training_gui(self):
        """
        Runs a process using a GUI to select files for training a sentiment classifier.
        This method uses unsupervised data (via PCA) to potentially improve
        features for a supervised Logistic Regression model trained on labeled data.
        It expects data in the libsvm/svmlight format often used in NLP datasets like IMDB.
        """

        def get_file(title="Select file") -> str:
            """Helper function to open a file dialog using Tkinter."""
            root = Tk() 
            root.withdraw() 
            
            path = filedialog.askopenfilename(title=title)
            root.update()
            # root.destroy() 
            return path

        # --- File Selection ---
        print("Select labeledBow.feat file...")
        labeled_path = get_file("Select labeledBow.feat")
        if not labeled_path: # Exit if no file selected
            print("No labeled data file selected. Exiting training.")
            return

        print("Select unsupBow.feat file...")
        unsup_path = get_file("Select unsupBow.feat")
        if not unsup_path: # Exit if no file selected
            print("No unsupervised data file selected. Exiting training.")
            return

        print("Select imdb.vocab file...")
        vocab_path = get_file("Select imdb.vocab")
        if not vocab_path: # Exit if no file selected
            print("No vocabulary file selected. Exiting training.")
            return

        # --- Data Loading Helper Functions ---
        def load_vocab(vocab_file: str) -> List[str]:
            """Loads vocabulary from a file (one word per line)."""
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f: # Added encoding
                    return [line.strip() for line in f]
            except FileNotFoundError:
                print(f"Vocabulary file not found: {vocab_path}")
                return []
            except Exception as e:
                print(f"Error reading vocabulary file {vocab_path}: {e}")
                return []

        def load_feat_file(filepath: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Loads labeled data from a .feat file (libsvm format variant).
            Assumes first element is a rating score (1-10), followed by index:count pairs.
            Converts ratings > 6 to 1 (positive), < 5 to 0 (negative), ignores 5-6.
            """
            data, labels = [], []
            try:
                with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
                    for line_num, line in enumerate(f):
                        try:
                            # Split line into parts and convert to integers
                            parts = list(map(int, line.strip().split()))
                            # Determine label based on the rating (first element)
                            rating = parts[0]
                            if rating > 6:
                                label = 1 # Positive
                            elif rating < 5:
                                label = 0 # Negative
                            else:
                                continue # Skip neutral reviews (rating 5 or 6)

                            # Create a sparse feature vector (initialized to zeros)
                            features = np.zeros(vocab_size)
                            # Populate feature vector using index:count pairs
                            # Starts from index 1, step 2 (index, count, index, count...)
                            for i in range(1, len(parts), 2):
                                feature_index = parts[i]
                                feature_count = parts[i+1]
                                if 0 <= feature_index < vocab_size: # Check index bounds
                                    features[feature_index] = feature_count
                                else:
                                    print(f"Warning: Feature index {feature_index} out of bounds (vocab size {vocab_size}) in {filepath}, line {line_num+1}. Skipping feature.")
                            data.append(features)
                            labels.append(label)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skipping malformed line {line_num+1} in {filepath}: {line.strip()} ({e})")
            except FileNotFoundError:
                 print(f"Feature file not found: {filepath}")
                 return np.array([]), np.array([]) # Return empty arrays
            except Exception as e:
                 print(f"Error reading feature file {filepath}: {e}")
                 return np.array([]), np.array([]) # Return empty arrays

            return np.array(data), np.array(labels)

        def load_unsup_feat(filepath: str, vocab_size: int) -> np.ndarray:
            """
            Loads unsupervised data from a .feat file (libsvm format variant).
            Assumes no label/rating at the start, just index:count pairs.
            """
            data = []
            try:
                with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
                    for line_num, line in enumerate(f):
                        try:
                            # Split line into parts and convert to integers
                            parts = list(map(int, line.strip().split()))
                            # Create a sparse feature vector
                            features = np.zeros(vocab_size)
                            # Populate feature vector using index:count pairs
                            # Starts from index 0, step 2 (index, count, index, count...)
                            for i in range(0, len(parts), 2):
                                feature_index = parts[i]
                                feature_count = parts[i+1]
                                if 0 <= feature_index < vocab_size: # Check index bounds
                                     features[feature_index] = feature_count
                                else:
                                    print(f"Warning: Feature index {feature_index} out of bounds (vocab size {vocab_size}) in {filepath}, line {line_num+1}. Skipping feature.")
                            data.append(features)
                        except (ValueError, IndexError) as e:
                             print(f"Warning: Skipping malformed line {line_num+1} in {filepath}: {line.strip()} ({e})")
            except FileNotFoundError:
                print(f"Unsupervised feature file not found: {filepath}")
                return np.array([]) # Return empty array
            except Exception as e:
                print(f"Error reading unsupervised feature file {filepath}: {e}")
                return np.array([]) # Return empty array
            return np.array(data)

        # --- Data Loading and Preprocessing ---
        vocab = load_vocab(vocab_path)
        if not vocab: # Exit if vocab loading failed
             print("Vocabulary is empty. Exiting training.")
             return
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")

        X_labeled, y_labeled = load_feat_file(labeled_path, vocab_size)
        if X_labeled.size == 0 or y_labeled.size == 0: # Check if loading failed
             print("Failed to load labeled data or data is empty. Exiting training.")
             return
        print(f"Loaded labeled data shape: {X_labeled.shape}")

        X_unsup = load_unsup_feat(unsup_path, vocab_size)
        if X_unsup.size == 0: # Check if loading failed
             print("Failed to load unsupervised data or data is empty. Exiting training.")
             return
        print(f"Loaded unsupervised data shape: {X_unsup.shape}")

        # --- PCA Dimension Reduction ---
        print("Fitting PCA on unsupervised data...")
        # Initialize PCA to reduce dimensions to 100
        # Using unsupervised data helps find general patterns in word usage
        pca = PCA(n_components=100, random_state=42) # Added random_state for reproducibility
        # Fit PCA on the larger unsupervised dataset
        pca.fit(X_unsup)
        print(f"PCA fitted. Explained variance ratio (top 100 components): {np.sum(pca.explained_variance_ratio_):.4f}")

        # --- Transform Labeled Data ---
        print("Transforming labeled data using fitted PCA...")
        # Apply the learned PCA transformation to the labeled feature set
        X_reduced = pca.transform(X_labeled)
        print(f"Transformed labeled data shape: {X_reduced.shape}")

        # --- Train/Test Split ---
        print("Splitting labeled data into train and test sets...")
        # Split the reduced labeled data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled # Added stratify
        )
        print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # --- Model Training ---
        print("Training logistic regression model...")
        # Initialize a Logistic Regression classifier
        clf = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter, added random_state
        # Train the classifier on the reduced-dimension training data
        clf.fit(X_train, y_train)
        print("Training complete.")

        # --- Evaluation ---
        print("Evaluating model on the test set...")
        # Make predictions on the test set
        predictions = clf.predict(X_test)
        # Print a detailed classification report (precision, recall, F1-score)
        print("\nClassification Report:\n")
        try:
            # Ensure there are predicted samples before generating report
            if len(predictions) > 0:
                 print(classification_report(y_test, predictions, target_names=['negative', 'positive']))
            else:
                 print("No predictions were made (Test set might be empty).")
        except ValueError as e:
            print(f"Could not generate classification report: {e}")



def main():
    """
    Main function to initialize and run the sentiment analysis process.
    """
    # TODO: Load a sentiment lexicon from a file or define it here.
    # Example: lexicon = {'good': 1.0, 'bad': -1.0, 'veri': 0.0, ...} # Stemmed words!
    # A common lexicon is AFINN, VADER, SentiWordNet (requires processing)
    # lexicon: Dict[str, float] = {    }
    # print(f"Using basic example lexicon with {len(lexicon)} entries.")

    # Initialize the analyzer with the lexicon
    # analyzer = MovieSentimentAnalyzer(lexicon)

    # Run the unsupervised training pipeline using GUI for file selection
    # analyzer.run_unsup_training_gui()

# Standard Python entry point check
if __name__ == '__main__':
    main()