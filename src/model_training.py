"""
Model training module for yield curve spread prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import logging
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/model_training.log'),
        logging.StreamHandler()
    ]
)

class TimeSeriesDataset(Dataset):
    """Dataset for LSTM model."""
    def __init__(self, features, targets, seq_length=10):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return x, y

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class MLPModel(nn.Module):
    """Multi-Layer Perceptron model for time series prediction."""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        
        # Create list of layer sizes including input and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Create layers dynamically
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Don't add activation/dropout after final layer
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """Base class for model training with walk-forward validation."""
    
    def __init__(self, spread: str, prediction_type: str, model_type: str):
        """
        Initialize ModelTrainer.
        
        Args:
            spread: Name of the spread ('2s10s', '5s30s', '2s5s', '10s30s', '3m10y')
            prediction_type: Type of prediction ('next_day', 'direction', 'ternary')
            model_type: Type of model ('ridge', 'lasso', 'rf', 'xgb', 'lstm', 'mlp', 'arima')
        """
        self.spread = spread
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()  # Initialize scaler
        self.features = None
        self.target = None
        self.results = {}
        
        # MLP hyperparameters
        if self.model_type == 'mlp':
            self.hidden_sizes = [512, 256, 128]  # Three hidden layers
            self.dropout = 0.2
            self.learning_rate = 0.001
            self.batch_size = 32
            self.epochs = 100
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.target_scaler = MinMaxScaler()  # For scaling regression targets
        
        # Create directories for saving models and results
        self.models_dir = Path('results/model_pickles')
        self.results_dir = Path('results/model_training')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized ModelTrainer for {spread} {prediction_type} with {model_type}")
    
    def load_selected_features(self) -> List[str]:
        """Load selected features from feature analysis results."""
        feature_file = Path(f'results/feature_analysis/y_{self.spread}_{self.prediction_type}/selected_features.csv')
        if not feature_file.exists():
            logging.error(f"Selected features file not found at {feature_file}")
            return None
        selected_features = pd.read_csv(feature_file)['0'].tolist()  # Use the second column with header '0'
        logging.info(f"Loaded {len(selected_features)} selected features")
        return selected_features
    
    def load_data(self):
        """Load and prepare the data with selected features."""
        try:
            # Load selected features
            selected_features = self.load_selected_features()
            if selected_features is None:
                return None, None
            
            # Load features and targets
            features_path = Path('data/processed/features.csv')
            targets_path = Path('data/processed/targets.csv')
            
            if not features_path.exists():
                logging.error(f"Features file not found at {features_path}")
                return None, None
            if not targets_path.exists():
                logging.error(f"Targets file not found at {targets_path}")
                return None, None
            
            # Load data with datetime index
            features = pd.read_csv(features_path, index_col=0)  # Use first column as index
            targets = pd.read_csv(targets_path, index_col=0)  # Use first column as index
            
            # Convert index to datetime
            features.index = pd.to_datetime(features.index)
            targets.index = pd.to_datetime(targets.index)
            
            # Filter features to only selected ones
            try:
                features = features[selected_features]
            except KeyError as e:
                logging.error(f"Some selected features not found in features data: {e}")
                return None, None
            
            # Get target column name
            target_col = f'y_{self.spread}_{self.prediction_type}'
            if target_col not in targets.columns:
                logging.error(f"Target column {target_col} not found in targets data")
                return None, None
            
            # Align features and targets
            common_index = features.index.intersection(targets.index)
            features = features.loc[common_index]
            target = targets.loc[common_index, target_col]
            
            # Scale features if using linear models
            if self.model_type in ['ridge', 'lasso']:
                features = pd.DataFrame(
                    self.scaler.fit_transform(features),
                    index=features.index,
                    columns=features.columns
                )
            
            self.features = features
            self.target = target
            
            logging.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
            logging.info(f"Date range: {features.index.min()} to {features.index.max()}")
            
            return features, target
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None, None
    
    def create_model(self) -> None:
        """Create model instance based on model_type."""
        if self.prediction_type == 'next_day':
            if self.model_type == 'ridge':
                self.model = Ridge(alpha=1.0)
            elif self.model_type == 'lasso':
                self.model = Lasso(alpha=1.0)
            elif self.model_type == 'rf':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                self.model = xgb.XGBRegressor(random_state=42)
        
        elif self.prediction_type in ['direction', 'ternary']:
            # Handle class imbalance
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.target),
                y=self.target
            )
            class_weight_dict = dict(zip(np.unique(self.target), class_weights))
            
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    class_weight=class_weight_dict,
                    random_state=42
                )
            elif self.model_type == 'xgb':
                if self.prediction_type == 'ternary':
                    self.model = xgb.XGBClassifier(
                        objective='multi:softmax',
                        num_class=3,
                        random_state=42
                    )
                else:
                    self.model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        random_state=42
                    )
        
        else:
            raise ValueError(f"Unknown target type: {self.prediction_type}")
        
        logging.info(f"Created {self.model_type} model for {self.prediction_type} prediction")
    
    def walk_forward_validation(self, n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            results = {
                'mse': [],
                'accuracy': [],
                'f1': [],
                'roc_auc': [],
                'feature_importance': [],
                'predictions': [],
                'actuals': []
            }
            
            # Convert to numpy arrays for sklearn compatibility
            X = self.features.values
            y = self.target.values
            
            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                if self.prediction_type == 'next_day':
                    y_pred = self.model.predict(X_val)
                    results['mse'].append(mean_squared_error(y_val, y_pred))
                else:
                    y_pred = self.model.predict(X_val)
                    y_prob = self.model.predict_proba(X_val)
                    results['accuracy'].append(accuracy_score(y_val, y_pred))
                    results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                    if self.prediction_type == 'direction':
                        results['roc_auc'].append(roc_auc_score(y_val, y_prob[:, 1]))
                
                # Store predictions and actuals
                results['predictions'].extend(y_pred)
                results['actuals'].extend(y_val)
                
                # Store feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    results['feature_importance'].append(
                        pd.Series(self.model.feature_importances_, index=self.features.columns)
                    )
            
            # Calculate average metrics
            self.results = {
                'mse': results['mse'] if results['mse'] else None,  # Store all MSE values
                'accuracy': np.mean(results['accuracy']) if results['accuracy'] else None,
                'f1': np.mean(results['f1']) if results['f1'] else None,
                'roc_auc': np.mean(results['roc_auc']) if results['roc_auc'] else None,
                'feature_importance': pd.concat(results['feature_importance'], axis=1).mean(axis=1) 
                    if results['feature_importance'] else None,
                'predictions': results['predictions'],
                'actuals': results['actuals']
            }
            
            logging.info(f"Completed walk-forward validation for {self.spread} {self.prediction_type}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in walk-forward validation: {str(e)}")
            return None
    
    def save_model(self) -> None:
        """Save trained model to disk."""
        model_file = self.models_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}.pkl"
        joblib.dump(self.model, model_file)
        logging.info(f"Saved model to {model_file}")
    
    def save_results(self, results: Dict) -> None:
        """Save training results with feature information."""
        results_dir = self.results_dir / f"{self.spread}_{self.prediction_type}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            'spread': self.spread,
            'prediction_type': self.prediction_type,
            'model_type': self.model_type,
            'num_features': len(self.features.columns),
            'selected_features': list(self.features.columns),
            'mse': results.get('mse'),
            'accuracy': results.get('accuracy'),
            'f1': results.get('f1'),
            'roc_auc': results.get('roc_auc'),
            'feature_importance': results.get('feature_importance', {}).to_dict() if isinstance(results.get('feature_importance'), pd.Series) else {}
        }
        
        # Save to JSON
        results_file = results_dir / f"{self.model_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Saved results to {results_file}")
    
    def _create_lstm_model(self) -> nn.Module:
        """Create LSTM model based on prediction type."""
        input_size = len(self.features.columns)
        hidden_size = 128
        num_layers = 2
        
        if self.prediction_type == 'next_day':
            output_size = 1
        elif self.prediction_type == 'direction':
            output_size = 2
        else:  # ternary
            output_size = 3
            
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        ).to(self.device)
        
        return model

    def _prepare_lstm_data(self, features: pd.DataFrame, target: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for LSTM model using walk-forward validation indices.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            train_idx: Training data indices
            val_idx: Validation data indices
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features.iloc[train_idx])
        scaled_features_val = self.scaler.transform(features.iloc[val_idx])
        
        # Scale target for regression
        if self.prediction_type == 'next_day':
            target_train = self.target_scaler.fit_transform(target.iloc[train_idx].values.reshape(-1, 1))
            target_train = target_train.flatten()
            target_val = self.target_scaler.transform(target.iloc[val_idx].values.reshape(-1, 1))
            target_val = target_val.flatten()
        else:
            target_train = target.iloc[train_idx].values
            target_val = target.iloc[val_idx].values
        
        # Create datasets
        train_dataset = TimeSeriesDataset(scaled_features, target_train, self.seq_length)
        val_dataset = TimeSeriesDataset(scaled_features_val, target_val, self.seq_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader

    def _train_lstm_epoch(self, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Train LSTM for one epoch."""
        model.train()
        total_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def _validate_lstm(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate LSTM model."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train_lstm(self) -> Dict:
        """Train LSTM model using walk-forward validation."""
        try:
            # Load data
            features, target = self.load_data()
            if features is None or target is None:
                return None
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            all_train_losses = []
            all_val_losses = []
            all_predictions = []
            all_actuals = []
            
            # Walk-forward validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                logging.info(f"Training fold {fold+1}/5")
                
                # Prepare data for this fold
                train_loader, val_loader = self._prepare_lstm_data(features, target, train_idx, val_idx)
                
                # Create model and optimizer
                model = self._create_lstm_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                
                # Set criterion based on prediction type
                if self.prediction_type == 'next_day':
                    criterion = nn.MSELoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                
                # Training loop
                best_val_loss = float('inf')
                patience = 5
                patience_counter = 0
                
                for epoch in range(self.epochs):
                    train_loss = self._train_lstm_epoch(model, train_loader, criterion, optimizer)
                    val_loss = self._validate_lstm(model, val_loader, criterion)
                    
                    logging.info(f"Fold {fold+1}, Epoch [{epoch+1}/{self.epochs}], "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model for this fold
                        torch.save(model.state_dict(), 
                                 self.models_dir / f"{self.spread}_{self.prediction_type}_lstm_fold{fold+1}.pth")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logging.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
                
                # Load best model for this fold
                model.load_state_dict(torch.load(
                    self.models_dir / f"{self.spread}_{self.prediction_type}_lstm_fold{fold+1}.pth"))
                
                # Get predictions for validation set
                model.eval()
                fold_predictions = []
                fold_actuals = []
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        outputs = model(batch_features)
                        
                        if self.prediction_type == 'next_day':
                            # Inverse transform predictions for regression
                            outputs = self.target_scaler.inverse_transform(
                                outputs.cpu().numpy().reshape(-1, 1)).flatten()
                        else:
                            # Get class predictions for classification
                            outputs = outputs.argmax(dim=1).cpu().numpy()
                        
                        fold_predictions.extend(outputs)
                        fold_actuals.extend(batch_targets.cpu().numpy())
                
                all_predictions.extend(fold_predictions)
                all_actuals.extend(fold_actuals)
                all_train_losses.append(train_loss)
                all_val_losses.append(val_loss)
            
            # Compute metrics based on prediction type
            if self.prediction_type == 'next_day':
                mse = mean_squared_error(all_actuals, all_predictions)
                results = {
                    'mse': mse,
                    'train_loss': np.mean(all_train_losses),
                    'val_loss': np.mean(all_val_losses),
                    'predictions': all_predictions,
                    'actuals': all_actuals
                }
            else:
                accuracy = accuracy_score(all_actuals, all_predictions)
                f1 = f1_score(all_actuals, all_predictions, average='weighted')
                results = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'train_loss': np.mean(all_train_losses),
                    'val_loss': np.mean(all_val_losses),
                    'predictions': all_predictions,
                    'actuals': all_actuals
                }
            
            logging.info(f"Completed walk-forward validation for LSTM")
            return results
            
        except Exception as e:
            logging.error(f"Error in LSTM training: {str(e)}")
            return None

    def _create_arima_model(self) -> ARIMA:
        """Create ARIMA model with optimal parameters."""
        try:
            # Use auto_arima to find optimal parameters
            auto_model = auto_arima(
                self.target,
                start_p=0, start_q=0, start_P=0, start_Q=0,
                max_p=5, max_q=5, max_P=5, max_Q=5,
                m=5,  # seasonal period
                seasonal=True,
                d=1, D=1,  # differencing
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            # Get optimal parameters
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            
            # Create ARIMA model with optimal parameters
            model = ARIMA(
                self.target,
                order=order,
                seasonal_order=seasonal_order
            )
            
            logging.info(f"Created ARIMA model with order={order}, seasonal_order={seasonal_order}")
            return model
            
        except Exception as e:
            logging.error(f"Error creating ARIMA model: {str(e)}")
            return None

    def train_arima(self) -> Dict:
        """Train ARIMA model using walk-forward validation."""
        try:
            # Load data
            features, target = self.load_data()
            if features is None or target is None:
                return None
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            all_predictions = []
            all_actuals = []
            all_mse = []
            
            # Walk-forward validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(target)):
                logging.info(f"Training fold {fold+1}/5")
                
                # Split data
                y_train = target.iloc[train_idx]
                y_val = target.iloc[val_idx]
                
                try:
                    # Create and fit model
                    model = self._create_arima_model()
                    if model is None:
                        continue
                        
                    results = model.fit()
                    
                    # Make predictions
                    predictions = results.forecast(steps=len(val_idx))
                    
                    # Store results
                    mse = mean_squared_error(y_val, predictions)
                    all_mse.append(mse)
                    all_predictions.extend(predictions)
                    all_actuals.extend(y_val)
                    
                    # Save model for this fold
                    model_path = self.models_dir / f"{self.spread}_{self.prediction_type}_arima_fold{fold+1}.pkl"
                    joblib.dump(results, model_path)
                    
                    logging.info(f"Fold {fold+1} MSE: {mse:.4f}")
                    
                except Exception as e:
                    logging.error(f"Error in fold {fold+1}: {str(e)}")
                    continue
            
            if not all_mse:
                logging.error("No successful folds in ARIMA training")
                return None
            
            # Compute final metrics
            results = {
                'mse': np.mean(all_mse),
                'predictions': all_predictions,
                'actuals': all_actuals,
                'model_type': 'arima'
            }
            
            logging.info(f"Completed ARIMA training with average MSE: {results['mse']:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"Error in ARIMA training: {str(e)}")
            return None

    def _create_mlp_model(self) -> nn.Module:
        """Create MLP model based on prediction type."""
        input_size = len(self.features.columns)
        
        if self.prediction_type == 'next_day':
            output_size = 1
        elif self.prediction_type == 'direction':
            output_size = 2
        else:  # ternary
            output_size = 3
            
        model = MLPModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=output_size,
            dropout=self.dropout
        ).to(self.device)
        
        return model

    def _prepare_mlp_data(self, features: pd.DataFrame, target: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for MLP model using walk-forward validation indices.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            train_idx: Training data indices
            val_idx: Validation data indices
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features.iloc[train_idx])
        scaled_features_val = self.scaler.transform(features.iloc[val_idx])
        
        # Scale target for regression
        if self.prediction_type == 'next_day':
            target_train = self.target_scaler.fit_transform(target.iloc[train_idx].values.reshape(-1, 1))
            target_train = target_train.flatten()
            target_val = self.target_scaler.transform(target.iloc[val_idx].values.reshape(-1, 1))
            target_val = target_val.flatten()
        else:
            target_train = target.iloc[train_idx].values
            target_val = target.iloc[val_idx].values
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(scaled_features)
        y_train = torch.FloatTensor(target_train)
        X_val = torch.FloatTensor(scaled_features_val)
        y_val = torch.FloatTensor(target_val)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader

    def _train_mlp_epoch(self, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Train MLP for one epoch."""
        model.train()
        total_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def _validate_mlp(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate MLP model."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train_mlp(self) -> Dict:
        """Train MLP model using walk-forward validation."""
        try:
            # Load data
            features, target = self.load_data()
            if features is None or target is None:
                return None
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            all_train_losses = []
            all_val_losses = []
            all_predictions = []
            all_actuals = []
            
            # Walk-forward validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                logging.info(f"Training fold {fold+1}/5")
                
                # Prepare data for this fold
                train_loader, val_loader = self._prepare_mlp_data(features, target, train_idx, val_idx)
                
                # Create model and optimizer
                model = self._create_mlp_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                
                # Set criterion based on prediction type
                if self.prediction_type == 'next_day':
                    criterion = nn.MSELoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                
                # Training loop
                best_val_loss = float('inf')
                patience = 5
                patience_counter = 0
                
                for epoch in range(self.epochs):
                    train_loss = self._train_mlp_epoch(model, train_loader, criterion, optimizer)
                    val_loss = self._validate_mlp(model, val_loader, criterion)
                    
                    logging.info(f"Fold {fold+1}, Epoch [{epoch+1}/{self.epochs}], "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model for this fold
                        torch.save(model.state_dict(), 
                                 self.models_dir / f"{self.spread}_{self.prediction_type}_mlp_fold{fold+1}.pth")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logging.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
                
                # Load best model for this fold
                model.load_state_dict(torch.load(
                    self.models_dir / f"{self.spread}_{self.prediction_type}_mlp_fold{fold+1}.pth"))
                
                # Get predictions for validation set
                model.eval()
                fold_predictions = []
                fold_actuals = []
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        outputs = model(batch_features)
                        
                        if self.prediction_type == 'next_day':
                            # Inverse transform predictions for regression
                            outputs = self.target_scaler.inverse_transform(
                                outputs.cpu().numpy().reshape(-1, 1)).flatten()
                        else:
                            # Get class predictions for classification
                            outputs = outputs.argmax(dim=1).cpu().numpy()
                        
                        fold_predictions.extend(outputs)
                        fold_actuals.extend(batch_targets.cpu().numpy())
                
                all_predictions.extend(fold_predictions)
                all_actuals.extend(fold_actuals)
                all_train_losses.append(train_loss)
                all_val_losses.append(val_loss)
            
            # Compute metrics based on prediction type
            if self.prediction_type == 'next_day':
                mse = mean_squared_error(all_actuals, all_predictions)
                results = {
                    'mse': mse,
                    'train_loss': np.mean(all_train_losses),
                    'val_loss': np.mean(all_val_losses),
                    'predictions': all_predictions,
                    'actuals': all_actuals
                }
            else:
                accuracy = accuracy_score(all_actuals, all_predictions)
                f1 = f1_score(all_actuals, all_predictions, average='weighted')
                results = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'train_loss': np.mean(all_train_losses),
                    'val_loss': np.mean(all_val_losses),
                    'predictions': all_predictions,
                    'actuals': all_actuals
                }
            
            logging.info(f"Completed walk-forward validation for MLP")
            return results
            
        except Exception as e:
            logging.error(f"Error in MLP training: {str(e)}")
            return None

    def train(self):
        """Train model and save results."""
        try:
            # Load data for all model types
            features, target = self.load_data()
            if features is None or target is None:
                return None
            
            # Train model based on type
            if self.model_type == 'lstm':
                results = self.train_lstm()
            elif self.model_type == 'mlp':
                results = self.train_mlp()
            elif self.model_type == 'arima':
                results = self.train_arima()
            elif self.model_type in ['ridge', 'lasso', 'rf', 'xgb']:
                self.create_model()
                results = self.walk_forward_validation()
            else:
                logging.error(f"Unsupported model type: {self.model_type}")
                return None
            
            if results is None:
                return None
            
            # Save results
            self.save_results(results)
            
            # Save model if not already saved (LSTM and MLP save per fold)
            if self.model_type in ['ridge', 'lasso', 'rf', 'xgb']:
                self.save_model()
            
            return results
            
        except Exception as e:
            logging.error(f"Error in training: {e}")
            return None

    @staticmethod
    def run_batch_training():
        """Run training for all spreads and prediction types."""
        # Define spreads and prediction types
        spreads = ['2s10s', '5s30s', '2s5s', '10s30s', '3m10y']
        prediction_types = ['next_day', 'direction', 'ternary']
        model_types = ['ridge', 'lasso', 'rf', 'xgb', 'lstm', 'mlp', 'arima']
        
        # Create results directories if they don't exist
        Path('results/model_training').mkdir(parents=True, exist_ok=True)
        Path('results/model_pickles').mkdir(parents=True, exist_ok=True)
        
        # Initialize results DataFrame
        results_summary = []
        
        # Run training for each combination
        for spread in spreads:
            for pred_type in prediction_types:
                # Skip ternary for 10s30s as discussed
                if spread == '10s30s' and pred_type == 'ternary':
                    logging.info(f"Skipping {spread} {pred_type} as previously discussed")
                    continue
                    
                for model_type in model_types:
                    # ARIMA only for next_day prediction
                    if model_type == 'arima' and pred_type != 'next_day':
                        logging.info(f"Skipping {model_type} for {pred_type} prediction")
                        continue
                        
                    try:
                        logging.info(f"Training {spread} {pred_type} with {model_type}")
                        
                        # Initialize trainer
                        trainer = ModelTrainer(
                            spread=spread,
                            prediction_type=pred_type,
                            model_type=model_type
                        )
                        
                        # Train model
                        results = trainer.train()
                        
                        if results is not None:
                            # Add to summary
                            summary = {
                                'spread': spread,
                                'prediction_type': pred_type,
                                'model_type': model_type,
                                'mse': results.get('mse'),
                                'accuracy': results.get('accuracy'),
                                'f1': results.get('f1'),
                                'roc_auc': results.get('roc_auc'),
                                'train_loss': results.get('train_loss'),
                                'val_loss': results.get('val_loss'),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            results_summary.append(summary)
                            
                            logging.info(f"Completed {spread} {pred_type} with {model_type}")
                            logging.info(f"Results: {summary}")
                        else:
                            logging.error(f"Training failed for {spread} {pred_type} with {model_type}")
                        
                    except Exception as e:
                        logging.error(f"Error training {spread} {pred_type} with {model_type}: {str(e)}")
        
        # Save summary results
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_df.to_csv('results/model_training/summary_results.csv', index=False)
            
            # Also save as JSON for better readability
            with open('results/model_training/summary_results.json', 'w') as f:
                json.dump(results_summary, f, indent=4)
            
            logging.info("Training completed. Summary saved to results/model_training/summary_results.csv and .json")
        else:
            logging.error("No successful training runs to summarize")

if __name__ == "__main__":
    ModelTrainer.run_batch_training() 