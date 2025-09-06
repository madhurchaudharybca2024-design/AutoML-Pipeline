from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, model, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        if len(X) < 2 or len(y) < 2:
            raise ValueError("Not enough data to split into train/test sets.")
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)