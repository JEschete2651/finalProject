
class Utility:
    def __init__(self):
        pass
    
    def load_feat(self, path):
    # Return X, y
        return

    def save_model(self, obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
