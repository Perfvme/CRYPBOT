from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self, model_path=None):
        try:
            self.model = RandomForestClassifier() if not model_path else joblib.load(model_path)
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}. Using default RandomForestClassifier.")
            self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        """Train the model."""
        logger.info(f"Training model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)

    def save_model(self, path):
        """Save the trained model."""
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully at {path}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def evaluate(self, X_test, y_test):
        """Evaluate model accuracy."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
