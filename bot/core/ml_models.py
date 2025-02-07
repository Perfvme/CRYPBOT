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
        """Evaluate model performance using multiple metrics."""
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
        """Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit."""

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
                                        scoring='f1',  # Or 'roc_auc'
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
        # Use RFECV for automatic cross-validated selection of the best number of features
        selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='f1', verbose=1, n_jobs=-1)
        selector = selector.fit(X_train, y_train)

        logger.info(f"Optimal number of features: {selector.n_features_}")
        logger.info(f"Best features (RFECV): {X_train.columns[selector.support_].tolist()}")
        self.selected_features = X_train.columns[selector.support_].tolist()
        return self.selected_features



    def backtest(self, df, symbol, initial_capital=10000, commission_pct=0.001):
        """
        Perform a simplified backtest, using the *preprocessed* DataFrame.

        Args:
            df: DataFrame containing *preprocessed* historical data (including features).
            symbol: The trading symbol (e.g., 'BTCUSDT').
            initial_capital: Starting capital.
            commission_pct: Commission percentage.

        Returns:
            Dictionary containing backtesting results.
        """

        # Filter the DataFrame to include only the selected features
        features_df = df[self.selected_features]

        # Initialize variables
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]

        for i in range(len(features_df)):
            # Get features for the current timestep
            current_features = features_df.iloc[[i]]

            # Predict probabilities and get confidence
            probabilities = self.predict_proba(current_features)[0]
            confidence = max(probabilities)
            prediction = self.predict(current_features)[0]

            # Generate signal based on prediction and confidence
            if confidence >= 0.8:
                signal = "BUY" if prediction == 1 else "SELL"
            else:
                signal = "HOLD"

            # Get current price data from the *original* DataFrame (not the features_df)
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]

            # Trading logic (rest of the backtesting logic remains the same)
            if signal == "BUY" and position <= 0:  # Buy (or close short)
                if position == -1:  # Close short position
                    profit = (entry_price - current_close) * abs(position) * capital
                    commission = commission_pct * (entry_price + current_close) * abs(position) * capital
                    capital += profit - commission
                    trades.append({
                        "type": "CLOSE_SHORT",
                        "price": current_close,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })

                position = 1  # Go long
                entry_price = current_close
                commission = commission_pct * entry_price * position * capital
                capital -= commission
                trades.append({
                    "type": "BUY",
                    "price": entry_price,
                    "profit": -commission,  # Commission is a loss
                    "timestamp": df.index[i]
                })

            elif signal == "SELL" and position >= 0:  # Sell (or close long)
                if position == 1:  # Close long position
                    profit = (current_close - entry_price) * position * capital
                    commission = commission_pct * (entry_price + current_close) * position * capital
                    capital += profit - commission
                    trades.append({
                        "type": "CLOSE_LONG",
                        "price": current_close,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })

                position = -1  # Go short
                entry_price = current_close
                commission = commission_pct * entry_price * abs(position) * capital
                capital -= commission
                trades.append({
                    "type": "SELL",
                    "price": entry_price,
                    "profit": -commission,
                    "timestamp": df.index[i]
                })

            # Stop-loss and take-profit (simplified)
            if position == 1:  # Long position
                stop_loss = entry_price * 0.98  # 2% stop-loss
                take_profit = entry_price * 1.05  # 5% take-profit
                if current_low <= stop_loss:
                    profit = (stop_loss - entry_price) * position * capital
                    commission = commission_pct * (entry_price + stop_loss) * position * capital
                    capital += profit - commission
                    trades.append({
                        "type": "STOP_LOSS",
                        "price": stop_loss,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })
                    position = 0
                elif current_high >= take_profit:
                    profit = (take_profit - entry_price) * position * capital
                    commission = commission_pct * (entry_price + take_profit) * position * capital
                    capital += profit - commission
                    trades.append({
                        "type": "TAKE_PROFIT",
                        "price": take_profit,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })
                    position = 0

            elif position == -1:  # Short position
                stop_loss = entry_price * 1.02  # 2% stop-loss
                take_profit = entry_price * 0.95  # 5% take-profit
                if current_high >= stop_loss:
                    profit = (entry_price - stop_loss) * abs(position) * capital
                    commission = commission_pct * (entry_price + stop_loss) * abs(position) * capital
                    capital += profit - commission
                    trades.append({
                        "type": "STOP_LOSS",
                        "price": stop_loss,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })
                    position = 0
                elif current_low <= take_profit:
                    profit = (entry_price - take_profit) * abs(position) * capital
                    commission = commission_pct * (entry_price + take_profit) * abs(position) * capital
                    capital += profit - commission
                    trades.append({
                        "type": "TAKE_PROFIT",
                        "price": take_profit,
                        "profit": profit - commission,
                        "timestamp": df.index[i]
                    })
                    position = 0

            equity_curve.append(capital)

        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            profitable_trades = trades_df[trades_df['profit'] > 0]
            win_rate = len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0
            average_profit = profitable_trades['profit'].mean() if not profitable_trades.empty else 0
            losing_trades = trades_df[trades_df['profit'] <= 0]
            average_loss = losing_trades['profit'].mean() if not losing_trades.empty else 0
            profit_factor = abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / trades_df[trades_df['profit'] <= 0]['profit'].sum()) if trades_df[trades_df['profit'] <= 0]['profit'].sum() != 0 else 0

            # Calculate maximum drawdown
            equity_series = pd.Series(equity_curve)
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()

            # Calculate Sharpe Ratio (simplified - assuming risk-free rate = 0)
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() !=0 else 0 # Annualized

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
            "final_capital": capital,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
            "equity_curve": equity_curve,
        }
