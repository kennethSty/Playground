from sklearn.model_selection import train_test_split

class PreProcessingEngine:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_split(self, test_size, random_state = 42):
        return train_test_split(self.x, self.y, test_size = test_size, random_state = random_state)
