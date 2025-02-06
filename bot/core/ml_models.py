from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
        if model_path:
            try:
                self.model = joblib.load(model_path)
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}. Using default RandomForestClassifier.")
                self.model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
        else:
            self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        """Train the model."""
        logger.info(f"Training model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Predict probabilities (for AUC)."""
        return self.model.predict_proba(X_test)

    def save_model(self, path):
        """Save the trained model."""
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully at {path}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using multiple metrics."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        # Handle potential errors with predict_proba (some models/hyperparameters might not support it)
        try:
            probabilities = self.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
            auc = roc_auc_score(y_test, probabilities)
        except:
            auc = None  # Set to None if predict_proba fails

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        if auc is not None:
            logger.info(f"AUC: {auc:.4f}")
        else:
            logger.info("AUC: Could not be calculated (predict_proba might not be supported).")

        return accuracy  # You can return any metric you want as the "primary" metric


    def tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters using GridSearchCV with TimeSeriesSplit."""
        param_grid = {
            'n_estimators': [50, 100, 200],  # Reduced number of estimators for speed
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced'] # Handle class imbalance
        }
        tscv = TimeSeriesSplit(n_splits=5)  # Use TimeSeriesSplit
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                    param_grid=param_grid,
                                    cv=tscv,
                                    scoring='f1',  # Use F1-score (or another appropriate metric)
                                    n_jobs=-1,  # Use all available cores
                                    verbose=2)  # Increase verbosity for debugging
        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")

        self.model = grid_search.best_estimator_
