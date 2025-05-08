
from typing import List, Tuple

class Lexicon_Analyzer:
    def __init__(self, lexicon, text_preprocessor, settings):
        self.lexicon = lexicon
        self.text_preprocessor = text_preprocessor
        self.settings = settings
    
    def compute_sentiment_score(self, tokens: List[Tuple[str, bool]]):
        if self.settings.key("log_score_compute") is True:
            print(f"Scoring tokens: {tokens}")
        
        score = 0.0
        
        for word, neg in tokens:
            base = self.lexicon.get(word, 0.0)
            if self.settings.key("log_score_compute") is True:
                print(f"  {word}: base={base}, negated={neg}")
            if neg:
                score += -base
            else:
                score += base
            if self.settings.key("log_score_compute") is True:
                print(f"Total score: {score}")
            
            return score
                
    def score(self, text):
        tokens = self.text_preprocessor.preprocess(text)
        negated_tokens = self.text_preprocessor.apply_negation_handling(tokens)
        return self.compute_sentiment_score(negated_tokens)