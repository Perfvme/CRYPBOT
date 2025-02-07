from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import RFE, RFECV  # Import RFECV
import joblib
import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self, model_path=None, model_type='random_forest'):
        self.model_type = model_type
        if model_path:
            try:
                self.model = joblib.load(model_path)
                # Infer model type from loaded model
                if isinstance(self.model, RandomForestClassifier):
                    self.model_type = 'random_forest'
                elif isinstance(self.model, GradientBoostingClassifier):
                    self.model_type = 'gradient_boosting'
                elif isinstance(self.model, LogisticRegression):
                    self.model_type = 'logistic_regression'
                # Check for and load selected_features
                if hasattr(self.model, 'selected_features'):
                    self.selected_features = self.model.selected_features
                else:  # Handle case where loaded model doesn't have it
                    self.selected_features = None # Will be set during retraining
                    logger.warning("Loaded model does not have 'selected_features' attribute.")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}.  Initializing new model.")
                self.model = self._initialize_model()
                self.selected_features = None
        else:
            self.model = self._initialize_model()
            self.selected_features = None

    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, solver='liblinear')
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")

    def train(self, X_train, y_train):
        """Train the model."""
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Predict probabilities (for AUC and confidence)."""
        return self.model.predict_proba(X_test)

    def save_model(self, path):
        """Save the trained model, including selected_features."""
        try:
            # Store selected_features *in* the model object before saving
            if self.selected_features is not None:
                self.model.selected_features = self.selected_features
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully at {path}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        try:
            probabilities = self.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)
        except:
            auc = None
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        if auc is not None:
            logger.info(f"AUC: {auc:.4f}")
        else:
            logger.info("AUC: Could not be calculated (predict_proba might not be supported).")
        return accuracy

    def tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters using RandomizedSearchCV."""
        if self.model_type == 'random_forest':
            param_grid = {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [5, 10, 15, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__class_weight': [None, 'balanced']
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'model__n_estimators': [50, 100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'logistic_regression':
            param_grid = {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': [None, 'balanced']
            }
        else:
            raise ValueError(f"Invalid model_type for tuning: {self.model_type}")

        tscv = TimeSeriesSplit(n_splits=5)
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', self.model)
        ])
        grid_search = RandomizedSearchCV(estimator=pipeline,
                                        param_distributions=param_grid,
                                        cv=tscv,
                                        scoring='f1',
                                        n_jobs=-1,
                                        verbose=2,
                                        n_iter=20)
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")
        self.model = grid_search.best_estimator_.named_steps['model']

    def select_features(self, X_train, y_train, n_features_to_select=30):
        """Perform Recursive Feature Elimination with cross-validation (RFECV)."""
        estimator = self._initialize_model()
        selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='f1', verbose=1, n_jobs=-1)
        selector = selector.fit(X_train, y_train)
        logger.info(f"Optimal number of features: {selector.n_features_}")
        logger.info(f"Best features (RFECV): {X_train.columns[selector.support_].tolist()}")
        self.selected_features = X_train.columns[selector.support_].tolist()
        return self.selected_features

    def backtest(self, df, symbol, initial_capital=10000, commission_pct=0.001):
        """Perform a simplified backtest using log returns."""

        features_df = df[self.selected_features]
        close_prices = df['close'].values  # Use NumPy arrays for speed
        log_capital = np.log(initial_capital)
        position = 0
        trades = []
        equity_curve = [initial_capital]

        # Pre-calculate log returns
        log_returns = np.log(close_prices[1:] / close_prices[:-1])

        for i in range(len(features_df) -1): # Iterate up to len - 1
            current_features = features_df.iloc[[i]]
            probabilities = self.predict_proba(current_features)[0]
            confidence = max(probabilities)
            prediction = self.predict(current_features)[0]

            if confidence >= 0.8:
                signal = "BUY" if prediction == 1 else "SELL"
            else:
                signal = "HOLD"

            # Use pre-calculated log return
            log_return = log_returns[i]

            if signal == "BUY" and position <= 0:
                if position == -1:
                    log_capital += -log_return  # Close short
                    # Commission: Apply to the *capital* used, not the price
                    log_capital -= commission_pct
                    trades.append({"type": "CLOSE_SHORT", "price": close_prices[i+1], "log_return": -log_return, "timestamp": df.index[i+1]})
                position = 1
                # Commission on entry
                log_capital -= commission_pct
                trades.append({"type": "BUY", "price": close_prices[i+1], "log_return": -commission_pct, "timestamp": df.index[i+1]})


            elif signal == "SELL" and position >= 0:
                if position == 1:
                    log_capital += log_return  # Close long
                    log_capital -= commission_pct
                    trades.append({"type": "CLOSE_LONG", "price": close_prices[i+1], "log_return": log_return, "timestamp": df.index[i+1]})
                position = -1
                log_capital -= commission_pct
                trades.append({"type": "SELL", "price": close_prices[i+1], "log_return": -commission_pct, "timestamp": df.index[i+1]})

            # Stop-loss and take-profit (log-based)
            if position != 0:  # Only check if we're in a position
                if position == 1:
                    # Calculate stop-loss and take-profit as *log returns* from the entry price
                    entry_log_price = np.log(trades[-1]['price'])  # Log price of the *last* trade
                    stop_loss_return = np.log(0.98)  # 2% below entry
                    take_profit_return = np.log(1.05) # 5% above entry
                    current_log_return = log_return # Log return since *last* period

                    if entry_log_price + current_log_return <= entry_log_price + stop_loss_return:
                        log_capital += stop_loss_return
                        log_capital -= commission_pct
                        trades.append({"type": "STOP_LOSS", "price": np.exp(entry_log_price + stop_loss_return), "log_return": stop_loss_return, "timestamp": df.index[i+1]})
                        position = 0
                    elif entry_log_price + current_log_return >= entry_log_price + take_profit_return:
                        log_capital += take_profit_return
                        log_capital -= commission_pct
                        trades.append({"type": "TAKE_PROFIT", "price": np.exp(entry_log_price + take_profit_return), "log_return": take_profit_return, "timestamp": df.index[i+1]})
                        position = 0

                elif position == -1:
                    entry_log_price = np.log(trades[-1]['price'])
                    stop_loss_return = np.log(1.02)  # 2% above entry
                    take_profit_return = np.log(0.95) # 5% below entry
                    current_log_return = log_return

                    if entry_log_price - current_log_return <= entry_log_price + take_profit_return :
                        log_capital -= current_log_return # Subtract, because short position
                        log_capital -= commission_pct
                        trades.append({"type": "TAKE_PROFIT", "price": np.exp(entry_log_price + take_profit_return), "log_return": -current_log_return, "timestamp": df.index[i+1]})
                        position = 0

                    elif entry_log_price - current_log_return >= entry_log_price + stop_loss_return:
                        log_capital -= current_log_return # Subtract log return (becomes addition)
                        log_capital -= commission_pct
                        trades.append({"type": "STOP_LOSS", "price": np.exp(entry_log_price+stop_loss_return), "log_return": -current_log_return, "timestamp": df.index[i+1]})
                        position = 0



            equity_curve.append(np.exp(log_capital))

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            profitable_trades = trades_df[trades_df['log_return'] > 0]
            win_rate = len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0
            average_profit = profitable_trades['log_return'].mean() if not profitable_trades.empty else 0
            losing_trades = trades_df[trades_df['log_return'] <= 0]
            average_loss = losing_trades['log_return'].mean() if not losing_trades.empty else 0
            profit_factor = abs(profitable_trades['log_return'].sum() / losing_trades['log_return'].sum()) if losing_trades['log_return'].sum() != 0 else 0

            equity_series = pd.Series(equity_curve)
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()

            returns = equity_series.pct_change().dropna()
            epsilon = 1e-8
            sharpe_ratio = returns.mean() / (returns.std() + epsilon) * np.sqrt(252) if returns.std() !=0 else 0

        else:
            win_rate = 0
            average_profit = 0
            average_loss = 0
            profit_factor = 0
            max_drawdown = 0
            sharpe_ratio = 0

        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": np.exp(log_capital),
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
            "equity_curve": equity_curve,
        }
