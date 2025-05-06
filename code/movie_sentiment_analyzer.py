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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV
)
from tkinter import filedialog, Tk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# ─── Settings Load/Save ───────────────────────────────────────────────────────

SETTINGS_FILE = "settings.json"

def load_settings() -> Dict:
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
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
            "log_settings_save": True,
            "cv_folds": 5,
            "grid_C": [0.01, 0.1, 1, 10]
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
    def __init__(self, lexicon: Dict[str, float]):
        self.lexicon = lexicon
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
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
        if settings.get("log_init", True):
            print("Sentiment analyzer initialized with lexicon.")

    def preprocess_text(self, text: str) -> List[str]:
        if settings.get("log_preprocess", True):
            print(f"Original text: {text}")
        text_lower = text.lower()
        if settings.get("log_preprocess", True):
            print(f"Lowercased text: {text_lower}")
        punctuation = string.punctuation.replace("'", "")
        cleaned = re.sub(f"[{re.escape(punctuation)}]", "", text_lower)
        if settings.get("log_preprocess", True):
            print(f"Punctuation removed: {cleaned}")
        tokens = cleaned.split()
        if settings.get("log_preprocess", True):
            print(f"Tokenized: {tokens}")
        lemmatized = [
            self.lemmatizer.lemmatize(w)
            for w in tokens
            if (w in self.negation_words) or (w not in self.stop_words)
        ]
        if settings.get("log_preprocess", True):
            print(f"Lemmatized tokens: {lemmatized}")
        return lemmatized

    def apply_negation_handling(self, tokens: List[str]) -> List[Tuple[str, bool]]:
        result, i = [], 0
        if settings.get("log_negation", True):
            print(f"Before negation: {tokens}")
        while i < len(tokens):
            if tokens[i] in self.negation_words:
                if settings.get("log_negation", True):
                    print(f"Negator '{tokens[i]}' found.")
                for j in range(i+1, min(i+1+self.negation_scope, len(tokens))):
                    result.append((tokens[j], True))
                i += self.negation_scope + 1
            else:
                result.append((tokens[i], False))
                i += 1
        if settings.get("log_negation", True):
            print(f"After negation: {result}")
        return result

    def compute_sentiment_score(self, toks: List[Tuple[str, bool]]) -> float:
        if settings.get("log_score_compute", True):
            print(f"Scoring tokens: {toks}")
        score = 0.0
        for word, neg in toks:
            base = self.lexicon.get(word, 0.0)
            if settings.get("log_score_compute", True):
                print(f"  {word}: base={base}, negated={neg}")
            score += -base if neg else base
        if settings.get("log_score_compute", True):
            print(f"Total score: {score}")
        return score

    def classify_sentiment(self, score: float) -> str:
        return "positive" if score >= 0 else "negative"

    def analyze_review(self, text: str) -> Tuple[str, float]:
        tokens = self.preprocess_text(text)
        negated = self.apply_negation_handling(tokens)
        score = self.compute_sentiment_score(negated)
        if settings.get("verbose_tokens", False):
            for w, neg in negated:
                val = self.lexicon.get(w, 0.0)
                print(f"{w} (neg={neg}) => {(-val if neg else val):.2f}")
        return self.classify_sentiment(score), score

    def train_pca_lr(self):
        # paths
        for key, title in [("pca_labeled","Select labeledBow.feat"),
                           ("pca_unsup","Select unsupBow.feat"),
                           ("pca_vocab","Select imdb.vocab")]:
            p = settings.get(key, "")
            if not p or not os.path.exists(p):
                if settings.get("log_pca_training_debug", True):
                    print(f"Requesting path for {key}")
                root=Tk(); root.withdraw()
                p = filedialog.askopenfilename(title=title)
                root.update()
                if not p:
                    print("Cancelled PCA+LR training."); return
                settings[key] = p; save_settings()
            elif settings.get("log_pca_training_debug", True):
                print(f"Using saved {key}: {p}")

        lab, uns, vocabf = settings["pca_labeled"], settings["pca_unsup"], settings["pca_vocab"]
        with open(vocabf, 'r', encoding='utf-8') as f:
            vocab_list = [w.strip() for w in f]
        vsz = len(vocab_list)
        Xl, yl = _load_feat(lab, vsz, True)
        Xu    = _load_feat(uns, vsz, False)

        self.pca = PCA(n_components=100, random_state=42)
        if settings.get("log_pca_training_debug", True):
            print("Fitting PCA on unsupervised data...")
        self.pca.fit(Xu)
        Xr = self.pca.transform(Xl)

        cv = settings.get("cv_folds", 5)
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(base_clf, Xr, yl, cv=cv)
        print(f"PCA+LR {cv}-fold CV acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid={"C": settings.get("grid_C", [0.01,0.1,1,10])},
            cv=cv, n_jobs=-1, verbose=0
        )
        grid.fit(Xr, yl)
        print(f"PCA+LR GridSearch best C: {grid.best_params_['C']}, mean CV acc: {grid.best_score_:.3f}")

        self.clf = grid.best_estimator_
        joblib.dump((self.pca, self.clf, vocab_list), "pca_lr_model.joblib")
        print("PCA+LR trained & saved to 'pca_lr_model.joblib'")

    def test_pca_lr(self):
        if not os.path.exists("pca_lr_model.joblib"):
            print("No PCA+LR model found. Train first."); return

        tf = settings.get("pca_test", "")
        if not tf or not os.path.exists(tf):
            if settings.get("log_pca_test_debug", True):
                print("Requesting test .feat file...")
            root=Tk(); root.withdraw()
            tf = filedialog.askopenfilename(title="Select labeledBow.feat for TEST")
            root.update()
            if not tf:
                print("Cancelled PCA+LR testing."); return
            settings["pca_test"] = tf; save_settings()

        self.pca, self.clf, vocab_list = joblib.load("pca_lr_model.joblib")
        vsz = len(vocab_list)
        X, y = _load_feat(settings["pca_test"], vsz, True)
        Xr = self.pca.transform(X)

        preds = self.clf.predict(Xr)
        probs = None
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(Xr)[:,1]

        print("\nPCA+LR Test Report:")
        print(classification_report(y, preds, target_names=['neg','pos']))

        cm = confusion_matrix(y, preds)
        print("Confusion Matrix:\n", cm)
        if probs is not None:
            try:
                auc = roc_auc_score(y, probs)
                print(f"ROC AUC: {auc:.3f}")
            except Exception as e:
                print(f"Could not compute AUC: {e}")

# ─── Directory-based supervised train/test ───────────────────────────────────

def train_directory_supervised():
    td = settings.get("dir_train","")
    if not td or not os.path.isdir(td):
        if settings.get("log_dir_train_debug", True):
            print("Requesting TRAIN directory via GUI.")
        root=Tk(); root.withdraw()
        td = filedialog.askdirectory(title="Select TRAIN directory")
        root.update()
        if not td:
            print("Cancelled."); return
        settings["dir_train"] = td; save_settings()
    elif settings.get("log_dir_train_debug", True):
        print(f"Using saved dir_train: {td}")

    texts, labels = [], []
    for lbl,val in [('neg',0),('pos',1)]:
        sub = os.path.join(td, lbl)
        if not os.path.isdir(sub): continue
        for fn in os.listdir(sub):
            if fn.endswith('.txt'):
                texts.append(open(os.path.join(sub,fn),encoding='utf-8').read())
                labels.append(val)

    vect = TfidfVectorizer(max_features=10000)
    X = vect.fit_transform(texts)

    cv = settings.get("cv_folds", 5)
    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(base_clf, X, labels, cv=cv)
    print(f"Dir-based {cv}-fold CV acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid={"C": settings.get("grid_C",[0.01,0.1,1,10])},
        cv=cv, n_jobs=-1, verbose=0
    )
    grid.fit(X, labels)
    print(f"Dir-based GridSearch best C: {grid.best_params_['C']}, mean CV acc: {grid.best_score_:.3f}")

    clf = grid.best_estimator_
    joblib.dump((vect, clf), "dir_sup_model.joblib")
    print("Directory-based model trained & saved to 'dir_sup_model.joblib'")

def test_directory_supervised():
    if not os.path.exists("dir_sup_model.joblib"):
        print("No directory-based model found. Train first."); return

    td = settings.get("dir_test","")
    if not td or not os.path.isdir(td):
        if settings.get("log_dir_test_debug", True):
            print("Requesting TEST directory via GUI.")
        root=Tk(); root.withdraw()
        td = filedialog.askdirectory(title="Select TEST directory")
        root.update()
        if not td:
            print("Cancelled."); return
        settings["dir_test"] = td; save_settings()
    elif settings.get("log_dir_test_debug", True):
        print(f"Using saved dir_test: {td}")

    vect, clf = joblib.load("dir_sup_model.joblib")
    texts, labels = [], []
    for lbl,val in [('neg',0),('pos',1)]:
        sub = os.path.join(td, lbl)
        if not os.path.isdir(sub): continue
        for fn in os.listdir(sub):
            if fn.endswith('.txt'):
                texts.append(open(os.path.join(sub,fn),encoding='utf-8').read())
                labels.append(val)

    X = vect.transform(texts)
    preds = clf.predict(X)
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:,1]

    print("\nDirectory-Based Test Report:")
    print(classification_report(labels, preds, target_names=['neg','pos']))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)
    if probs is not None:
        try:
            auc = roc_auc_score(labels, probs)
            print(f"ROC AUC: {auc:.3f}")
        except Exception as e:
            print(f"Could not compute AUC: {e}")

# ─── Lexicon loader ──────────────────────────────────────────────────────────

def select_and_load_lexicon() -> Dict[str,float]:
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
            print("No file selected."); return {}
        settings["lexicon_path"] = lp; save_settings()
    elif settings.get("log_lexicon_debug", True):
        print(f"Using saved lexicon_path: {lp}")

    try:
        df = pd.read_csv(lp)
    except Exception as e:
        print(f"Error reading lexicon: {e}"); return {}

    if 'stemmed_word' not in df.columns or 'sentiment_score' not in df.columns:
        print("CSV must have 'stemmed_word' and 'sentiment_score' columns."); return {}

    lmtzr = WordNetLemmatizer()
    lex = {
        lmtzr.lemmatize(str(w)): float(s)
        for w, s in zip(df['stemmed_word'], df['sentiment_score'])
    }
    print(f"Loaded lexicon with {len(lex)} entries.")
    return lex

# ─── Feature loader helper ───────────────────────────────────────────────────

def _load_feat(path: str, vocab_size: int, labeled: bool):
    X, y = [], ([] if labeled else None)
    for ln in open(path, 'r', encoding='utf-8'):
        toks = ln.split()
        if not toks: continue
        if labeled:
            try:
                r = int(toks[0])
            except:
                continue
            lab = 1 if r>6 else 0 if r<5 else None
            if lab is None:
                continue
            feats = toks[1:]
        else:
            feats = toks if ':' in toks[0] else toks[1:]
        vec = np.zeros(vocab_size)
        for t in feats:
            if ':' in t:
                i, c = map(int, t.split(':'))
                if 0 <= i < vocab_size:
                    vec[i] = c
        X.append(vec)
        if labeled:
            y.append(lab)
    X = np.array(X)
    return (X, np.array(y)) if labeled else X

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
                print("3. Load new lexicon")
                print("4. Back to main menu")
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
                        print("File not found."); continue
                    with open(path,'r',encoding='utf-8') as f:
                        review = f.read()
                    lbl, sc = analyzer.analyze_review(review)
                    print(f"Sentiment: {lbl}, Score: {sc:.2f}")
                elif sub == '3':
                    new_lex = select_and_load_lexicon()
                    if new_lex:
                        analyzer.lexicon = new_lex
                        print("New lexicon loaded successfully.")
                elif sub == '4':
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
