import json

class Settings:
    def __init__(self):
        self.path = "settings.json"
        
        pass
    
    # Loads the settings from the settings json file
    def load_settings(self):
        try:
            with open(self.path, 'r') as f:
                settings = json.load(f)
        except FileExistsError:
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

    
    # Saves user settings to the settings json file
    def save_settings(self, settings):
        with open(self.path, "w") as f:
            json.dump(settings, f, indent=4)
    
    def key_exist(self, key):
        settings = self.load_settings()
        if settings.get(key, True):
            return True
        else:
            return False
        