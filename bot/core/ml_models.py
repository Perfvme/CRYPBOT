from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class MLModel:
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier() if not model_path else joblib.load(model_path)

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)

    def save_model(self, path):
        """Save the trained model."""
        joblib.dump(self.model, path)

    def evaluate(self, X_test, y_test):
        """Evaluate model accuracy."""
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)