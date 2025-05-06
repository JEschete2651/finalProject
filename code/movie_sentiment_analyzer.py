import os
import re
import string
import json
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tkinter import filedialog, Tk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ─── Settings Load/Save ───────────────────────────────────────────────────────

SETTINGS_FILE = "settings.json"

def load_settings() -> Dict:
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        # defaults
        settings = {
            "lexicon_path": "",
            "pca_labeled": "",
            "pca_unsup": "",
            "pca_vocab": "",
            "pca_test": "",
            "dir_train": "",
            "dir_test": "",
            "verbose_tokens": False,
            "log_init": True,
            "log_preprocess": True,
            "log_negation": True,
            "log_score_compute": True,
            "log_lexicon_debug": True,
            "log_pca_training_debug": True,
            "log_pca_test_debug": True,
            "log_dir_train_debug": True,
            "log_dir_test_debug": True,
            "log_settings_load": True,
            "log_settings_save": True
        }
    return settings

settings = load_settings()
if settings.get("log_settings_load", True):
    print(f"Settings loaded from {SETTINGS_FILE}: {settings}")

def save_settings():
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    if settings.get("log_settings_save", True):
        print(f"Settings saved to {SETTINGS_FILE}")

# ─── MovieSentimentAnalyzer ───────────────────────────────────────────────────

class MovieSentimentAnalyzer:
    """
    A class to perform sentiment analysis on movie reviews using a lexicon-based approach
    and optionally train a supervised model using features derived from unsupervised data.
    """
    def __init__(self, lexicon: Dict[str, float]):
        self.lexicon = lexicon
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()             # retained for backward compatibility
        self.lemmatizer = WordNetLemmatizer()      # new lemmatizer for improved accuracy
        self.negation_words = {
            "not", "no", "never", "none", "n't", "without",
            "neither", "nor", "hardly", "barely", "scarcely",
            "don't", "doesn't", "isn't", "wasn't", "weren't",
            "can't", "won't", "shouldn't", "couldn't", "wouldn't"
        }
        self.negation_scope = 3
        # placeholders for PCA+LR
        self.pca = None
        self.clf = None
        if settings.get("log_init", True):
            print("Sentiment analyzer initialized with lexicon.")

    def preprocess_text(self, text: str) -> List[str]:
        """
        Lowercase, remove punctuation (except apostrophes), tokenize,
        remove stopwords (but preserve negators), and lemmatize.
        """
        if settings.get("log_preprocess", True):
            print(f"Original text: {text}")
        text_lower = text.lower()
        if settings.get("log_preprocess", True):
            print(f"Lowercased text: {text_lower}")
        # remove all punctuation except apostrophes so "don't" remains intact
        punctuation = string.punctuation.replace("'", "")
        cleaned_text = re.sub(f"[{re.escape(punctuation)}]", "", text_lower)
        if settings.get("log_preprocess", True):
            print(f"Removed punctuation (except apostrophes): {cleaned_text}")
        tokens = cleaned_text.split()
        if settings.get("log_preprocess", True):
            print(f"Tokenized text: {tokens}")
        # Lemmatize and filter out stopwords, but always keep negation words
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(w)
            for w in tokens
            if (w in self.negation_words) or (w not in self.stop_words)
        ]
        if settings.get("log_preprocess", True):
            print(f"Lemmatized tokens (stopwords removed, negators kept): {lemmatized_tokens}")
        return lemmatized_tokens

    def apply_negation_handling(self, tokens: List[str]) -> List[Tuple[str, bool]]:
        result = []
        i = 0
        if settings.get("log_negation", True):
            print(f"Tokens before negation handling: {tokens}")
        while i < len(tokens):
            if tokens[i] in self.negation_words:
                if settings.get("log_negation", True):
                    print(f"Negation word '{tokens[i]}' detected. Negating next {self.negation_scope} tokens.")
                for j in range(i + 1, min(i + 1 + self.negation_scope, len(tokens))):
                    result.append((tokens[j], True))
                i += self.negation_scope + 1
            else:
                result.append((tokens[i], False))
                i += 1
        if settings.get("log_negation", True):
            print(f"Tokens after negation handling: {result}")
        return result

    def compute_sentiment_score(self, tokens_with_negation: List[Tuple[str, bool]]) -> float:
        if settings.get("log_score_compute", True):
            print(f"Computing sentiment score for tokens: {tokens_with_negation}")
        score = 0.0
        for word, neg in tokens_with_negation:
            base = self.lexicon.get(word, 0.0)
            if settings.get("log_score_compute", True):
                print(f"Word '{word}': base score {base}, negated: {neg}")
            score += -base if neg else base
        if settings.get("log_score_compute", True):
            print(f"Total sentiment score: {score}")
        return score

    def classify_sentiment(self, score: float) -> str:
        return "positive" if score >= 0 else "negative"

    def analyze_review(self, text: str) -> Tuple[str, float]:
        tokens = self.preprocess_text(text)
        tokens_neg = self.apply_negation_handling(tokens)
        score = self.compute_sentiment_score(tokens_neg)

        # only print token-level debug if verbose_tokens==True
        if settings.get("verbose_tokens", False):
            for token, is_neg in tokens_neg:
                base = self.lexicon.get(token, 0.0)
                if is_neg:
                    base *= -1
                print(f"{token} (negated: {is_neg}) => {base}")

        label = self.classify_sentiment(score)
        return label, score

    def run_unsup_training_gui(self):
        # your original unsupervised + PCA + LR
        def get_file(title="Select file") -> str:
            root = Tk(); root.withdraw()
            path = filedialog.askopenfilename(title=title)
            root.update()
            return path

        print("Select labeledBow.feat...")
        labeled_path = get_file("Select labeledBow.feat")
        if not labeled_path:
            print("No labeled data selected."); return
        print("Select unsupBow.feat...")
        unsup_path = get_file("Select unsupBow.feat")
        if not unsup_path:
            print("No unsupervised data selected."); return
        print("Select imdb.vocab...")
        vocab_path = get_file("Select imdb.vocab")
        if not vocab_path:
            print("No vocab selected."); return

        # ... rest remains your original code ...

    def train_pca_lr(self):
        """Train PCA+LR on labeled+unsup, save both model+vocab, update settings."""
        for key, title in [("pca_labeled","Select labeledBow.feat"),
                           ("pca_unsup","Select unsupBow.feat"),
                           ("pca_vocab","Select imdb.vocab")]:
            p = settings.get(key,"")
            if not p or not os.path.exists(p):
                if settings.get("log_pca_training_debug", True):
                    print(f"Requesting path for {key} via GUI.")
                root=Tk(); root.withdraw()
                p = filedialog.askopenfilename(title=title)
                root.update()
                if not p:
                    print("Training cancelled.")
                    return
                settings[key] = p
                save_settings()
            else:
                if settings.get("log_pca_training_debug", True):
                    print(f"Using saved {key}: {p}")

        lab, uns, vocabf = settings["pca_labeled"], settings["pca_unsup"], settings["pca_vocab"]
        if settings.get("log_pca_training_debug", True):
            print("Loading vocabulary file...")
        with open(vocabf,'r',encoding='utf-8') as f:
            vocab_list = [w.strip() for w in f]
        vsz = len(vocab_list)
        if settings.get("log_pca_training_debug", True):
            print(f"Vocabulary size: {vsz}")

        def load_lab(path):
            X,y = [],[]
            for ln in open(path,'r',encoding='utf-8'):
                toks=ln.split()
                if not toks: continue
                try: r=int(toks[0])
                except: continue
                labl = 1 if r>6 else 0 if r<5 else None
                if labl is None: continue
                vec = np.zeros(vsz)
                for t in toks[1:]:
                    if ':' in t:
                        i,c=map(int,t.split(':'))
                        if 0<=i<vsz: vec[i]=c
                X.append(vec); y.append(labl)
            return np.array(X), np.array(y)

        def load_uns(path):
            X=[]
            for ln in open(path,'r',encoding='utf-8'):
                toks=ln.split()
                if not toks: continue
                if ':' not in toks[0]: toks=toks[1:]
                vec=np.zeros(vsz)
                for t in toks:
                    if ':' in t:
                        i,c=map(int,t.split(':'))
                        if 0<=i<vsz: vec[i]=c
                X.append(vec)
            return np.array(X)

        if settings.get("log_pca_training_debug", True):
            print("Loading labeled data...")
        Xl, yl = load_lab(lab)
        if settings.get("log_pca_training_debug", True):
            print(f"Labeled data loaded: {Xl.shape[0]} samples")
        Xu = load_uns(uns)
        if settings.get("log_pca_training_debug", True):
            print(f"Unsupervised data loaded: {Xu.shape[0]} samples")

        self.pca = PCA(n_components=100, random_state=42)
        if settings.get("log_pca_training_debug", True):
            print("Fitting PCA on unsupervised data...")
        self.pca.fit(Xu)
        if settings.get("log_pca_training_debug", True):
            print("PCA fit complete. Transforming labeled data...")
        Xr = self.pca.transform(Xl)

        self.clf = LogisticRegression(max_iter=1000, random_state=42)
        if settings.get("log_pca_training_debug", True):
            print("Training Logistic Regression classifier...")
        self.clf.fit(Xr, yl)
        if settings.get("log_pca_training_debug", True):
            print("Classifier training complete.")

        joblib.dump((self.pca, self.clf, vocab_list), "pca_lr_model.joblib")
        print("PCA+LR trained and saved to 'pca_lr_model.joblib'")

    def test_pca_lr(self):
        """Load PCA+LR model and run on a labeledBow.feat for test (remember path)."""
        if not os.path.exists("pca_lr_model.joblib"):
            print("No PCA+LR model found. Train first.")
            return

        tf = settings.get("pca_test","")
        if not tf or not os.path.exists(tf):
            if settings.get("log_pca_test_debug", True):
                print("Requesting test file for PCA+LR via GUI.")
            root=Tk(); root.withdraw()
            tf = filedialog.askopenfilename(title="Select labeledBow.feat for TEST")
            root.update()
            if not tf:
                print("Testing cancelled.")
                return
            settings["pca_test"] = tf
            save_settings()
        else:
            if settings.get("log_pca_test_debug", True):
                print(f"Using saved pca_test: {tf}")

        self.pca, self.clf, vocab_list = joblib.load("pca_lr_model.joblib")
        if settings.get("log_pca_test_debug", True):
            print("Model loaded. Preparing test data...")
        vsz = len(vocab_list)

        X, y = [], []
        for ln in open(tf, 'r', encoding='utf-8'):
            toks = ln.split()
            if not toks: continue
            try:
                r = int(toks[0])
            except:
                continue
            labl = 1 if r > 6 else 0 if r < 5 else None
            if labl is None: continue
            vec = np.zeros(vsz)
            for t in toks[1:]:
                if ':' in t:
                    i, c = map(int, t.split(':'))
                    if 0 <= i < vsz:
                        vec[i] = c
            X.append(vec)
            y.append(labl)


        Xr = self.pca.transform(np.array(X))
        if settings.get("log_pca_test_debug", True):
            print("Test data transformed. Running predictions...")
        preds = self.clf.predict(Xr)
        print("\nPCA+LR Test Report:")
        print(classification_report(y, preds, target_names=['neg','pos']))

# ─── Directory-based supervised train/test ───────────────────────────────────

def train_directory_supervised():
    """Train on a TRAIN dir (neg/pos), save model, remember path."""
    td = settings.get("dir_train","")
    if not td or not os.path.isdir(td):
        if settings.get("log_dir_train_debug", True):
            print("Requesting TRAIN directory via GUI.")
        root=Tk(); root.withdraw()
        td = filedialog.askdirectory(title="Select TRAIN directory")
        root.update()
        if not td:
            print("Cancelled.")
            return
        settings["dir_train"] = td
        save_settings()
    else:
        if settings.get("log_dir_train_debug", True):
            print(f"Using saved dir_train: {td}")

    texts, labels = [], []
    for lbl,val in [('neg',0),('pos',1)]:
        sub=os.path.join(td,lbl)
        if not os.path.isdir(sub): continue
        for fn in os.listdir(sub):
            if fn.endswith('.txt'):
                texts.append(open(os.path.join(sub,fn),encoding='utf-8').read())
                labels.append(val)

    vect = CountVectorizer(max_features=10000)
    X = vect.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, labels)

    joblib.dump((vect, clf), "dir_sup_model.joblib")
    print("Directory-based model trained & saved to 'dir_sup_model.joblib'")

def test_directory_supervised():
    """Load dir_sup model, test on TEST dir (neg/pos), remember path."""
    if not os.path.exists("dir_sup_model.joblib"):
        print("No directory-based model found. Train first.")
        return

    td = settings.get("dir_test","")
    if not td or not os.path.isdir(td):
        if settings.get("log_dir_test_debug", True):
            print("Requesting TEST directory via GUI.")
        root=Tk(); root.withdraw()
        td = filedialog.askdirectory(title="Select TEST directory")
        root.update()
        if not td:
            print("Cancelled.")
            return
        settings["dir_test"] = td
        save_settings()
    else:
        if settings.get("log_dir_test_debug", True):
            print(f"Using saved dir_test: {td}")

    vect, clf = joblib.load("dir_sup_model.joblib")
    texts, labels = [], []
    for lbl,val in [('neg',0),('pos',1)]:
        sub=os.path.join(td,lbl)
        if not os.path.isdir(sub): continue
        for fn in os.listdir(sub):
            if fn.endswith('.txt'):
                texts.append(open(os.path.join(sub,fn),encoding='utf-8').read())
                labels.append(val)

    X = vect.transform(texts)
    preds = clf.predict(X)
    print("\nDirectory-Based Test Report:")
    print(classification_report(labels, preds, target_names=['neg','pos']))

# ─── Lexicon loader (uses settings) ──────────────────────────────────────────

def select_and_load_lexicon() -> Dict[str,float]:
    """
    If a lexicon_path is saved, use it; otherwise ask once and remember it.
    """
    lp = settings.get("lexicon_path","")
    if not lp or not os.path.exists(lp):
        if settings.get("log_lexicon_debug", True):
            print("Prompting for lexicon file via GUI.")
        root=Tk(); root.withdraw()
        lp = filedialog.askopenfilename(
            title="Select Lexicon CSV", filetypes=[("CSV files","*.csv")]
        )
        root.update()
        if not lp:
            print("No file selected.")
            return {}
        settings["lexicon_path"] = lp
        save_settings()
    else:
        if settings.get("log_lexicon_debug", True):
            print(f"Using saved lexicon_path: {lp}")

    df = pd.read_csv(lp)
    if 'stemmed_word' not in df.columns or 'sentiment_score' not in df.columns:
        print("CSV must have 'stemmed_word' and 'sentiment_score' columns.")
        return {}
    lex = dict(zip(df['stemmed_word'].astype(str), df['sentiment_score'].astype(float)))
    print(f"Loaded lexicon with {len(lex)} entries.")
    return lex

# ─── Main Menu ───────────────────────────────────────────────────────────────

def main():
    analyzer = None

    while True:
        print("\n=== Sentiment Analyzer Menu ===")
        print("1. Lexicon-based analysis")
        print("2. PCA + LR model")
        print("3. Directory-based model")
        print("4. Settings")
        print("5. Exit")
        choice = input("Choose an option: ").strip()

        if choice == '1':
            if analyzer is None:
                lex = select_and_load_lexicon()
                if not lex:
                    continue
                analyzer = MovieSentimentAnalyzer(lex)
            while True:
                print("\n-- Lexicon-Based Submenu --")
                print("1. Analyze single review (manual input)")
                print("2. Analyze review from text file")
                print("3. Back to main menu")
                sub = input("Choose: ").strip()
                if sub == '1':
                    txt = input("Enter your review:\n> ")
                    lbl, sc = analyzer.analyze_review(txt)
                    print(f"Sentiment: {lbl}, Score: {sc:.2f}")
                elif sub == '2':
                    root=Tk(); root.withdraw()
                    path = filedialog.askopenfilename(
                        title="Select Review Text File", filetypes=[("Text files","*.txt")]
                    )
                    root.update()
                    if not path or not os.path.exists(path):
                        print("File not found.")
                        continue
                    with open(path,'r',encoding='utf-8') as f:
                        review = f.read()
                    lbl, sc = analyzer.analyze_review(review)
                    print(f"Sentiment: {lbl}, Score: {sc:.2f}")
                elif sub == '3':
                    break
                else:
                    print("Invalid choice.")

        elif choice == '2':
            print("\n-- PCA + LR Submenu --")
            print("1. Train PCA+LR model")
            print("2. Test PCA+LR model")
            sub = input("Choose: ").strip()
            if sub == '1':
                if analyzer is None:
                    analyzer = MovieSentimentAnalyzer({})
                analyzer.train_pca_lr()
            elif sub == '2':
                if analyzer is None:
                    analyzer = MovieSentimentAnalyzer({})
                analyzer.test_pca_lr()
            else:
                print("Invalid choice.")

        elif choice == '3':
            print("\n-- Directory-Based Submenu --")
            print("1. Train directory-based model")
            print("2. Test directory-based model")
            sub = input("Choose: ").strip()
            if sub == '1':
                train_directory_supervised()
            elif sub == '2':
                test_directory_supervised()
            else:
                print("Invalid choice.")

        elif choice == '4':
            # Settings submenu
            toggle_options = [
                ("verbose_tokens", "Verbose token-level output"),
                ("log_init", "Initialization messages"),
                ("log_preprocess", "Text preprocessing debug"),
                ("log_negation", "Negation handling debug"),
                ("log_score_compute", "Score computation debug"),
                ("log_lexicon_debug", "Lexicon loading debug"),
                ("log_pca_training_debug", "PCA training debug"),
                ("log_pca_test_debug", "PCA testing debug"),
                ("log_dir_train_debug", "Directory training debug"),
                ("log_dir_test_debug", "Directory testing debug"),
                ("log_settings_load", "Settings load debug"),
                ("log_settings_save", "Settings save debug")
            ]
            while True:
                print("\n-- Settings --")
                for idx, (key, desc) in enumerate(toggle_options, start=1):
                    state = "ON" if settings.get(key, True) else "OFF"
                    print(f"{idx}. Toggle {desc} (currently {state})")
                print(f"{len(toggle_options)+1}. Clear all saved paths")
                print(f"{len(toggle_options)+2}. Back to main menu")
                sub = input("Choose: ").strip()
                if sub.isdigit():
                    idx = int(sub)
                    if 1 <= idx <= len(toggle_options):
                        key, desc = toggle_options[idx-1]
                        settings[key] = not settings.get(key, True)
                        save_settings()
                        print(f"{desc} now {'ON' if settings[key] else 'OFF'}")
                    elif idx == len(toggle_options)+1:
                        for k in ["lexicon_path","pca_labeled","pca_unsup","pca_vocab","pca_test","dir_train","dir_test"]:
                            settings[k] = ""
                        save_settings()
                        print("Cleared all saved file paths.")
                    elif idx == len(toggle_options)+2:
                        break
                    else:
                        print("Invalid choice.")
                else:
                    print("Invalid choice.")

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid selection.")

if __name__ == '__main__':
    main()
