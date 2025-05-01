"""
Model training module for yield curve spread prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import logging
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from torch.cuda.amp import autocast, GradScaler
import pynvml
import traceback
import warnings

# Configure logging at the top level
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/model_training.log'),
        logging.StreamHandler()
    ]
)

# Create module-level logger
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (dict, list)):
            # Recursively handle nested dictionaries and lists
            if isinstance(obj, dict):
                return {k: self.default(v) for k, v in obj.items()}
            else:
                return [self.default(v) for v in obj]
        return super(NumpyEncoder, self).default(obj)

def ensure_json_serializable(obj):
    """Helper function to ensure an object is JSON serializable."""
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {k: ensure_json_serializable(v) for k, v in obj.items()}
        else:
            return [ensure_json_serializable(v) for v in obj]
    return obj

warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    """Dataset for LSTM model."""
    def __init__(self, features, targets, seq_length=10, classification: bool = False):
        # Convert features to numpy array if it's a pandas DataFrame/Series
        if hasattr(features, 'values'):
            features = features.values
        if hasattr(targets, 'values'):
            targets = targets.values
            
        # Ensure data is numeric and handle missing values
        try:
            features = np.array(features, dtype=np.float32)
            if classification:
                targets = np.array(targets, dtype=np.int64)  # Use int64 for classification
            else:
                targets = np.array(targets, dtype=np.float32)  # Use float32 for regression
        except (ValueError, TypeError) as e:
            logging.error(f"Error converting data to appropriate type: {str(e)}")
            # Try to handle non-numeric data by converting to appropriate type
            features = pd.DataFrame(features).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
            if classification:
                targets = pd.Series(targets).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
            else:
                targets = pd.Series(targets).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
            
        # Fill missing values with 0
        features = np.nan_to_num(features, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
            
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
            
        # Convert to appropriate tensors
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets) if classification else torch.FloatTensor(targets)
        self.seq_length = seq_length
        
    def __len__(self):
        return max(0, len(self.features) - self.seq_length)
        
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
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
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

class HyperparameterTuner:
    """Class to handle hyperparameter tuning for different model types."""
    
    @staticmethod
    def get_param_grid(model_type: str) -> Dict:
        """Get hyperparameter grid for specified model type."""
        if model_type == 'ridge':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif model_type == 'lasso':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif model_type == 'rf':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'xgb':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def tune_traditional_model(model_type: str, X: np.ndarray, y: np.ndarray, cv: TimeSeriesSplit) -> Tuple[object, Dict]:
        """Tune traditional ML models (Ridge, Lasso, RF, XGB) using TimeSeriesSplit CV."""
        param_grid = HyperparameterTuner.get_param_grid(model_type)
        
        if model_type == 'ridge':
            base_model = Ridge()
        elif model_type == 'lasso':
            base_model = Lasso()
        elif model_type == 'rf':
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == 'xgb':
            base_model = xgb.XGBRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported model type for traditional tuning: {model_type}")
        
        grid_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=8,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_
    
    @staticmethod
    def tune_arima(y: np.ndarray) -> Tuple[Dict, float]:
        """Tune ARIMA model using auto_arima."""
        try:
            model = auto_arima(
                y,
                start_p=0,
                start_q=0,
                max_p=4,
                max_q=4,
                m=1,
                start_P=0,
                seasonal=False,
                d=1,
                D=1,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            best_params = {
                'p': model.order[0],
                'd': model.order[1],
                'q': model.order[2]
            }
            
            return best_params, model.aic()
            
        except Exception as e:
            logger.error(f"Error in ARIMA tuning: {str(e)}")
            return None, None
    
    @staticmethod
    def tune_deep_learning(model_type: str, features: pd.DataFrame, target: pd.Series, 
                          train_idx: np.ndarray, val_idx: np.ndarray) -> Dict:
        """Tune deep learning models (MLP, LSTM) using validation set."""
        best_params = None
        global_best_val_loss = float('inf')
        
        # Create parameter combinations with reduced search space
        param_combinations = []
        if model_type == 'lstm':
            # Focus on most important parameters first
            param_combinations = [
                {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'sequence_length': 63},
                {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'sequence_length': 63},
                {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'sequence_length': 63}
            ]
        else:
            # For MLP, use a similar approach
            param_combinations = [
                {'hidden_sizes': [512, 256, 128], 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
                {'hidden_sizes': [256, 128, 64], 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64},
                {'hidden_sizes': [512, 256, 128], 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32}
            ]
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()
        
        for params in param_combinations:
            # Reset best_val_loss for each parameter combination
            combo_best_val_loss = float('inf')
            try:
                if model_type == 'mlp':
                    model = MLPModel(
                        input_size=len(features.columns),
                        hidden_sizes=params['hidden_sizes'],
                        output_size=1,  # For regression
                        dropout=params['dropout']
                    )
                elif model_type == 'lstm':
                    model = LSTMModel(
                        input_size=len(features.columns),
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        output_size=1,  # For regression
                        dropout=params['dropout']
                    )
                
                # Move model to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                
                # Train model with current parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                criterion = nn.MSELoss()
                
                if model_type == 'lstm':
                    # Use TimeSeriesDataset to create sequences for LSTM input
                    train_dataset = TimeSeriesDataset(
                        features.iloc[train_idx].astype(np.float32),
                        target.iloc[train_idx].astype(np.float32),
                        seq_length=params['sequence_length']
                    )
                    val_dataset = TimeSeriesDataset(
                        features.iloc[val_idx].astype(np.float32),
                        target.iloc[val_idx].astype(np.float32),
                        seq_length=params['sequence_length']
                    )
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
                else:
                    # For non-sequence models like MLP, do the normal conversion:
                    X_train = torch.FloatTensor(features.iloc[train_idx].astype(np.float32).values)
                    y_train = torch.FloatTensor(target.iloc[train_idx].astype(np.float32).values)
                    X_val   = torch.FloatTensor(features.iloc[val_idx].astype(np.float32).values)
                    y_val   = torch.FloatTensor(target.iloc[val_idx].astype(np.float32).values)
                    train_dataset = TensorDataset(X_train, y_train)
                    val_dataset = TensorDataset(X_val, y_val)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
                
                # Training loop with reduced epochs for tuning
                patience = 3  # Reduced patience for faster tuning
                patience_counter = 0
                best_model_state = None
                
                for epoch in range(20):  # Reduced max epochs for tuning
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad()
                        
                        # Use mixed precision training
                        with autocast():
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                        
                        # Scale loss and backpropagate using scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    
                    # Validation
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device)
                            outputs = model(batch_X)
                            val_loss += criterion(outputs, batch_y).item()
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < combo_best_val_loss:
                        combo_best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                
                # Update global best if this parameter combination is better
                if combo_best_val_loss < global_best_val_loss:
                    global_best_val_loss = combo_best_val_loss
                    best_params = params
                    logger.info(f"New best parameters found: {params} with val_loss: {global_best_val_loss:.4f}")
                
            except Exception as e:
                logger.error(f"Error in deep learning tuning with params {params}: {str(e)}")
                continue
        
        return best_params

class ModelTrainer:
    """Base class for model training with walk-forward validation."""
    
    def __init__(self, spread: str, prediction_type: str, model_type: str, tune_hyperparameters: bool = True):
        """
        Initialize ModelTrainer.
        
        Args:
            spread: Name of the spread ('2s10s', '5s30s', '2s5s', '10s30s', '3m10y')
            prediction_type: Type of prediction ('next_day', 'direction', 'ternary')
            model_type: Type of model ('ridge', 'lasso', 'rf', 'xgb', 'lstm', 'mlp', 'arima')
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        # Initialize logger for this instance
        self.logger = logging.getLogger(f'ModelTrainer_{spread}_{prediction_type}_{model_type}')
        
        self.spread = spread
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.model = None
        self.features = None
        self.target = None
        self.results = {}
        self.best_params = None
        self.epochs = 100  # Add epochs parameter
        
        # Enhanced device detection and logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"PyTorch Version: {torch.__version__}")
            
            # Get GPU utilization using helper method
            gpu_stats = self._get_gpu_utilization()
            if gpu_stats:
                self.logger.info(f"GPU utilization: {gpu_stats['gpu_util']}%")
                self.logger.info(f"GPU memory utilization: {gpu_stats['memory_util']}%")
                self.logger.info(f"GPU memory allocated: {gpu_stats['memory_allocated']:.2f} GB")
        else:
            self.logger.warning("No GPU available. Training will be performed on CPU.")
            self.logger.warning("This will significantly increase training time.")
            self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Initialize gradient scaler for mixed precision training
        self.amp_scaler = GradScaler()
        
        # Set paths
        self.data_dir = Path('data/processed')
        self.models_dir = Path('models')
        self.results_dir = Path('results/model_training')
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default hyperparameters
        self.set_default_hyperparameters()
        
        # Initialize model dimensions
        self.input_size = None  # Will be set when features are loaded
        self.output_size = None  # Will be set when target is loaded
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.logger.info(f"Initialized ModelTrainer for {spread} {prediction_type} with {model_type}")
    
    def set_default_hyperparameters(self):
        """Set default hyperparameters based on model type."""
        if self.model_type == 'mlp':
            self.hidden_sizes = [512, 256, 128]
            self.dropout = 0.2
            self.learning_rate = 0.001
            self.batch_size = 32
        elif self.model_type == 'lstm':
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.2
            self.learning_rate = 0.001
            self.batch_size = 32
            self.sequence_length = 63
        elif self.model_type == 'arima':
            self.best_params = None
        else:
            self.best_params = None
    
    def load_selected_features(self):
        """Load selected features from file."""
        try:
            features_file = Path(f'results/feature_analysis/y_{self.spread}_{self.prediction_type}/selected_features.csv')
            if not features_file.exists():
                self.logger.warning(f"Selected features file not found: {features_file}")
                return None
            
            # Read the CSV and get the feature names from the second column
            df = pd.read_csv(features_file, header=0)
            if len(df.columns) < 2:
                self.logger.error(f"Selected features file has incorrect format: {features_file}")
                return None
                
            selected_features = df.iloc[:, 1].tolist()
            self.logger.info(f"Loaded {len(selected_features)} selected features")
            return selected_features
        except Exception as e:
            self.logger.error(f"Error loading selected features: {str(e)}")
            return None
    
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
                self.logger.error(f"Features file not found at {features_path}")
                return None, None
            if not targets_path.exists():
                self.logger.error(f"Targets file not found at {targets_path}")
                return None, None
            
            # Load data with datetime index
            features = pd.read_csv(features_path, index_col=0)  # Use first column as index
            targets = pd.read_csv(targets_path, index_col=0)  # Use first column as index
            
            # Convert index to datetime
            features.index = pd.to_datetime(features.index)
            targets.index = pd.to_datetime(targets.index)
            
            # Ensure numeric data
            features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
            targets = targets.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Filter features to only selected ones
            try:
                # Convert feature names to match the format in features.csv
                selected_features_clean = [f.strip() for f in selected_features]
                available_features = features.columns.tolist()
                
                # Find matching features
                matching_features = []
                for feature in selected_features_clean:
                    # Look for exact matches first
                    if feature in available_features:
                        matching_features.append(feature)
                    else:
                        # If no exact match, look for features that start with the selected feature name
                        matches = [col for col in available_features if col.startswith(feature + '_')]
                        if matches:
                            matching_features.extend(matches)
                        else:
                            self.logger.warning(f"No matches found for feature: {feature}")
                
                if not matching_features:
                    self.logger.error("No matching features found in features data")
                    return None, None
                    
                features = features[matching_features]
                self.logger.info(f"Using {len(matching_features)} features")
                
            except KeyError as e:
                self.logger.error(f"Error selecting features: {e}")
                return None, None
            
            # Get target column name
            target_col = f'y_{self.spread}_{self.prediction_type}'
            if target_col not in targets.columns:
                self.logger.error(f"Target column {target_col} not found in targets data")
                return None, None
            
            # Align features and targets
            common_index = features.index.intersection(targets.index)
            features = features.loc[common_index]
            target = targets.loc[common_index, target_col]
            
            # For classification tasks, remap target values and update output_size accordingly
            if self.prediction_type in ['direction', 'ternary']:
                # Get unique values and create mapping
                unique_vals = np.unique(target)
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                
                # Log the mapping for debugging
                self.logger.info(f"Classification mapping for {self.prediction_type}: {mapping}")
                
                # Remap the targets
                target = target.map(mapping)
                
                # Set output_size to number of classes
                self.output_size = len(mapping)
                self.logger.info(f"Set output_size to {self.output_size} for {self.prediction_type} classification")
            else:
                # For regression, output_size is 1
                self.output_size = 1
            
            self.features = features
            self.target = target
            
            # Set input size
            self.input_size = len(features.columns)
            
            self.logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
            self.logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
            self.logger.info(f"Target range: {target.min()} to {target.max()}")
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error(traceback.format_exc())
            return None, None
    
    def create_model(self) -> None:
        """Create model instance based on model_type and hyperparameters."""
        if self.prediction_type == 'next_day':
            if self.model_type == 'ridge':
                params = self.best_params if self.best_params else {'alpha': 1.0}
                self.model = Ridge(**params)
            elif self.model_type == 'lasso':
                params = self.best_params if self.best_params else {'alpha': 1.0}
                self.model = Lasso(**params)
            elif self.model_type == 'rf':
                params = self.best_params if self.best_params else {
                    'n_estimators': 100,
                    'random_state': 42
                }
                self.model = RandomForestRegressor(**params)
            elif self.model_type == 'xgb':
                params = self.best_params if self.best_params else {
                    'random_state': 42
                }
                self.model = xgb.XGBRegressor(**params)
        
        elif self.prediction_type in ['direction', 'ternary']:
            # Handle class imbalance
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.target),
                y=self.target
            )
            class_weight_dict = dict(zip(np.unique(self.target), class_weights))
            
            if self.model_type == 'rf':
                params = self.best_params if self.best_params else {
                    'n_estimators': 100,
                    'class_weight': class_weight_dict,
                    'random_state': 42
                }
                self.model = RandomForestClassifier(**params)
            elif self.model_type == 'xgb':
                params = self.best_params if self.best_params else {
                    'random_state': 42
                }
                if self.prediction_type == 'ternary':
                    params.update({
                        'objective': 'multi:softmax',
                        'num_class': 3
                    })
                else:
                    params.update({
                        'objective': 'binary:logistic'
                    })
                self.model = xgb.XGBClassifier(**params)
        
        else:
            raise ValueError(f"Unknown target type: {self.prediction_type}")
        
        self.logger.info(f"Created {self.model_type} model for {self.prediction_type} prediction with params: {self.best_params}")
    
    def walk_forward_validation(self, n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Ensure model is created if not already
            if self.model is None:
                self.create_model()
                if self.model is None:
                    raise ValueError("Failed to create model for walk-forward validation")
            
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
                
                # Create and fit scaler on training data only
                feature_scaler = StandardScaler()
                X_train_scaled = feature_scaler.fit_transform(X_train)
                X_val_scaled = feature_scaler.transform(X_val)
                
                # Scale targets if needed (for regression)
                if self.prediction_type == 'next_day':
                    target_scaler = StandardScaler()
                    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
                    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
                else:
                    y_train_scaled = y_train
                    y_val_scaled = y_val
                
                # Create new model instance for this fold
                self.create_model()
                if self.model is None:
                    raise ValueError("Failed to create model for fold")
                
                # Train model
                self.model.fit(X_train_scaled, y_train_scaled)
                
                # Make predictions
                if self.prediction_type == 'next_day':
                    y_pred_scaled = self.model.predict(X_val_scaled)
                    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    results['mse'].append(mean_squared_error(y_val, y_pred))
                else:
                    y_pred = self.model.predict(X_val_scaled)
                    y_prob = self.model.predict_proba(X_val_scaled)
                    results['accuracy'].append(accuracy_score(y_val, y_pred))
                    results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                    if self.prediction_type == 'direction':
                        # Binary classification
                        results['roc_auc'].append(roc_auc_score(y_val, y_prob[:, 1]))
                    elif self.prediction_type == 'ternary':
                        # Multi-class classification
                        try:
                            results['roc_auc'].append(roc_auc_score(y_val, y_prob, multi_class='ovo'))
                        except Exception as e:
                            self.logger.warning(f"Could not compute ROC-AUC for ternary classification: {str(e)}")
                            results['roc_auc'].append(None)
                
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
                'mse': results['mse'] if results['mse'] else None,
                'accuracy': np.mean(results['accuracy']) if results['accuracy'] else None,
                'f1': np.mean(results['f1']) if results['f1'] else None,
                'roc_auc': np.mean(results['roc_auc']) if results['roc_auc'] else None,
                'feature_importance': pd.concat(results['feature_importance'], axis=1).mean(axis=1) 
                    if results['feature_importance'] else None,
                'predictions': results['predictions'],
                'actuals': results['actuals']
            }
            
            self.logger.info(f"Completed walk-forward validation for {self.spread} {self.prediction_type}")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward validation: {str(e)}")
            raise  # Re-raise to prevent silent failures
    
    def save_model(self) -> None:
        """Save trained model to disk."""
        model_file = self.models_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}.pkl"
        joblib.dump(self.model, model_file)
        self.logger.info(f"Saved model to {model_file}")
    
    def save_results(self, results: Dict) -> None:
        """Save training results with feature information and predictions."""
        results_dir = self.results_dir / f"{self.spread}_{self.prediction_type}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions to CSV if available
        if 'predictions' in results and 'actuals' in results:
            # Convert actuals to regular floats if they are in list format
            actuals = [float(val[0]) if isinstance(val, list) else float(val) for val in results['actuals']]
            predictions_df = pd.DataFrame({
                'prediction': results['predictions'],
                'actual': actuals
            })
            
            # Get dates from the data directory
            data_dir = Path('data/processed')
            dates_file = data_dir / f"{self.spread}_data.csv"
            if dates_file.exists():
                dates_df = pd.read_csv(dates_file)
                if len(dates_df) >= len(predictions_df):
                    # Use dates from the data file
                    predictions_df['date'] = dates_df['date'].iloc[-len(predictions_df):].values
                else:
                    # Create dates starting from the last available date
                    last_date = dates_df['date'].iloc[-1]
                    dates = pd.date_range(start=pd.to_datetime(last_date), periods=len(predictions_df))
                    predictions_df['date'] = dates.strftime('%Y-%m-%d')
            else:
                # Create default dates if no data file exists
                dates = pd.date_range(start='2010-01-01', periods=len(predictions_df))
                predictions_df['date'] = dates.strftime('%Y-%m-%d')
            
            # Save predictions to CSV
            predictions_file = results_dir / f"{self.model_type}_predictions.csv"
            predictions_df.to_csv(predictions_file, index=False)
            self.logger.info(f"Saved predictions to {predictions_file}")
        
        # Prepare metrics dictionary and ensure all values are JSON serializable
        metrics = {
            'spread': self.spread,
            'prediction_type': self.prediction_type,
            'model_type': self.model_type,
            'num_features': int(len(self.features.columns)),
            'selected_features': list(self.features.columns),
            'mse': ensure_json_serializable(results.get('mse')),
            'accuracy': ensure_json_serializable(results.get('accuracy')),
            'f1': ensure_json_serializable(results.get('f1')),
            'roc_auc': ensure_json_serializable(results.get('roc_auc')),
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional metrics if they exist
        if 'train_loss' in results:
            metrics['train_loss'] = ensure_json_serializable(results['train_loss'])
        if 'val_loss' in results:
            metrics['val_loss'] = ensure_json_serializable(results['val_loss'])
        if 'best_epoch' in results:
            metrics['best_epoch'] = ensure_json_serializable(results['best_epoch'])
        if 'training_time' in results:
            metrics['training_time'] = ensure_json_serializable(results['training_time'])
        
        # Handle feature importance separately to ensure proper serialization
        fi = results.get('feature_importance')
        if fi is not None:
            metrics['feature_importance'] = ensure_json_serializable(fi)
        else:
            metrics['feature_importance'] = {}

        # Save to JSON using the custom encoder
        results_file = results_dir / f"{self.model_type}_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=4, cls=NumpyEncoder)
            self.logger.info(f"Saved results to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """
        Get GPU utilization and memory usage.
        
        Returns:
            Dictionary with GPU utilization and memory usage percentages
        """
        try:
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return {
                    'gpu_util': util.gpu,
                    'memory_util': util.memory,
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3  # GB
                }
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            self.logger.warning(f"Could not get GPU utilization: {e}")
            return None

    def _create_lstm_model(self) -> nn.Module:
        """Create LSTM model with proper device placement."""
        try:
            model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                dropout=self.dropout
            )
            
            # Move model to device and log memory usage
            model = model.to(self.device)
            if torch.cuda.is_available():
                gpu_stats = self._get_gpu_utilization()
                if gpu_stats:
                    self.logger.info(f"Model moved to GPU. Current GPU memory usage: {gpu_stats['memory_allocated']:.2f} GB")
                    self.logger.info(f"GPU utilization: {gpu_stats['gpu_util']}%")
            
            return model
        except Exception as e:
            self.logger.error(f"Error creating LSTM model: {str(e)}")
            raise

    def _prepare_lstm_data(self, features: pd.DataFrame, target: pd.Series, 
                         train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        try:
            # Scale features
            X_train = features.iloc[train_idx].astype(np.float32)
            X_val = features.iloc[val_idx].astype(np.float32)
            
            # Fit scaler on training data only
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            if self.prediction_type == 'next_day':
                # Scale targets for regression
                y_train = target.iloc[train_idx].values.reshape(-1, 1).astype(np.float32)
                y_val = target.iloc[val_idx].values.reshape(-1, 1).astype(np.float32)
                
                # Fit target scaler on training data only
                y_train_scaled = self.target_scaler.fit_transform(y_train)
                y_val_scaled = self.target_scaler.transform(y_val)
                
                classification = False
            else:
                # For classification, convert targets to integers
                y_train_scaled = target.iloc[train_idx].values.astype(np.int64)
                y_val_scaled = target.iloc[val_idx].values.astype(np.int64)
                classification = True
            
            train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, self.sequence_length, classification)
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, self.sequence_length, classification)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True)
            
            return train_loader, val_loader
        except Exception as e:
            self.logger.error(f"Error in LSTM data preparation: {str(e)}")
            raise  # Re-raise to prevent silent failures

    def _train_lstm_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train LSTM model for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        try:
            if torch.cuda.is_available():
                gpu_stats = self._get_gpu_utilization()
                if gpu_stats:
                    self.logger.info(f"Start of epoch - GPU memory usage: {gpu_stats['memory_allocated']:.2f} GB")
                    self.logger.info(f"GPU utilization: {gpu_stats['gpu_util']}%  |  GPU memory utilization: {gpu_stats['memory_util']}%")
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                with autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                # Scale loss and backpropagate using amp_scaler
                self.amp_scaler.scale(loss).backward()
                
                # Gradient clipping
                self.amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                self.amp_scaler.step(optimizer)
                self.amp_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item()
                num_batches += 1
            
            # Handle case where no batches were processed
            if num_batches == 0:
                self.logger.warning("No batches processed in epoch - returning 0.0 loss")
                return 0.0
                
            return total_loss / num_batches
            
        except Exception as e:
            self.logger.error(f"Error in training epoch: {str(e)}")
            raise

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
                self.logger.error("Failed to load data for LSTM training")
                return None
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            all_train_losses = []
            all_val_losses = []
            all_predictions = []
            all_actuals = []
            
            # Walk-forward validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                self.logger.info(f"Starting fold {fold+1}/5 of walk-forward validation")
                
                # Prepare data for this fold
                train_loader, val_loader = self._prepare_lstm_data(features, target, train_idx, val_idx)
                if train_loader is None or val_loader is None:
                    self.logger.error(f"Failed to prepare data for fold {fold+1}")
                    continue
                
                try:
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
                    best_model_state = None
                    
                    for epoch in range(self.epochs):
                        self.logger.info(f"Starting epoch {epoch+1}/{self.epochs} for fold {fold+1}")
                        train_loss = self._train_lstm_epoch(model, train_loader, optimizer, criterion)
                        val_loss = self._validate_lstm(model, val_loader, criterion)
                        
                        self.logger.info(f"Fold {fold+1}, Epoch [{epoch+1}/{self.epochs}], "
                                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                        
                        # Early stopping
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_model_state = model.state_dict()
                            self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                                break
                    
                    # Load best model for this fold
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        self.logger.info(f"Loaded best model state for fold {fold+1}")
                    
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
                    
                    self.logger.info(f"Completed fold {fold+1}/5 with train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in fold {fold+1} training: {str(e)}")
                    continue
            
            if not all_train_losses:
                self.logger.error("No successful folds in LSTM training")
                return None
            
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
            
            self.logger.info(f"Completed walk-forward validation for LSTM with final metrics: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in LSTM training: {str(e)}")
            return None

    def _create_arima_model(self) -> ARIMA:
        """Create ARIMA model with optimal parameters."""
        try:
            # Determine seasonal period based on data frequency
            if self.features.index.freq == 'D':
                seasonal_period = 5  # Weekly pattern
            elif self.features.index.freq == 'W':
                seasonal_period = 4  # Monthly pattern
            elif self.features.index.freq == 'M':
                seasonal_period = 12  # Yearly pattern
            else:
                seasonal_period = 1  # No seasonality
            
            # Use tuned parameters if available, otherwise use auto_arima
            if self.best_params:
                order = (self.best_params['p'], self.best_params['d'], self.best_params['q'])
                self.logger.info(f"Using tuned ARIMA parameters: order={order}")
            else:
                # Fallback to auto_arima if no tuned parameters
                auto_model = auto_arima(
                    self.target,
                    start_p=0, start_q=0, start_P=0, start_Q=0,
                    max_p=5, max_q=5, max_P=5, max_Q=5,
                    m=seasonal_period,
                    seasonal=True,
                    d=1, D=1,  # differencing
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                order = auto_model.order
                self.logger.info(f"Using auto-selected ARIMA parameters: order={order}")
            
            # Create ARIMA model with determined parameters
            model = ARIMA(
                self.target,
                order=order,
                seasonal_order=(1, 1, 1, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Created ARIMA model with order={order}, seasonal_period={seasonal_period}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating ARIMA model: {str(e)}")
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
                self.logger.info(f"Training fold {fold+1}/5")
                
                # Split data
                y_train = target.iloc[train_idx]
                y_val = target.iloc[val_idx]
                
                try:
                    # Determine seasonal period based on data frequency
                    if self.features.index.freq == 'D':
                        seasonal_period = 5  # Weekly pattern
                    elif self.features.index.freq == 'W':
                        seasonal_period = 4  # Monthly pattern
                    elif self.features.index.freq == 'M':
                        seasonal_period = 12  # Yearly pattern
                    else:
                        seasonal_period = 1  # No seasonality
                    
                    # Use tuned parameters if available, otherwise use auto_arima
                    if self.best_params:
                        order = (self.best_params['p'], self.best_params['d'], self.best_params['q'])
                        self.logger.info(f"Using tuned ARIMA parameters: order={order}")
                    else:
                        # Fallback to auto_arima if no tuned parameters
                        auto_model = auto_arima(
                            y_train,
                            start_p=0, start_q=0, start_P=0, start_Q=0,
                            max_p=5, max_q=5, max_P=5, max_Q=5,
                            m=seasonal_period,
                            seasonal=True,
                            d=1, D=1,  # differencing
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True
                        )
                        order = auto_model.order
                        self.logger.info(f"Using auto-selected ARIMA parameters: order={order}")
                    
                    # Initialize history with training data
                    history = list(y_train)
                    fold_preds = []
                    
                    # Rolling one-step-ahead forecast
                    for true_val in y_val:
                        # Fit a fresh ARIMA on all data so far
                        model = ARIMA(
                            history,
                            order=order,
                            seasonal_order=(1, 1, 1, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0)
                        )
                        res = model.fit()
                        
                        # Forecast one step ahead
                        yhat = res.forecast(steps=1)[0]
                        fold_preds.append(yhat)
                        
                        # Roll the window forward by adding the real next value
                        history.append(true_val)
                    
                    # Store results
                    mse = mean_squared_error(y_val, fold_preds)
                    all_mse.append(mse)
                    all_predictions.extend(fold_preds)
                    all_actuals.extend(y_val)
                    
                    # Save model for this fold
                    model_path = self.models_dir / f"{self.spread}_{self.prediction_type}_arima_fold{fold+1}.pkl"
                    joblib.dump(res, model_path)
                    
                    self.logger.info(f"Fold {fold+1} MSE: {mse:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in fold {fold+1}: {str(e)}")
                    continue
            
            if not all_mse:
                self.logger.error("No successful folds in ARIMA training")
                return None
            
            # Compute final metrics
            results = {
                'mse': np.mean(all_mse),
                'predictions': all_predictions,
                'actuals': all_actuals,
                'model_type': 'arima'
            }
            
            self.logger.info(f"Completed ARIMA training with average MSE: {results['mse']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA training: {str(e)}")
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

    def _prepare_mlp_data(self, features: pd.DataFrame, target: pd.Series, 
                          train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        try:
            # Scale features
            X_train = features.iloc[train_idx].astype(np.float32)
            X_val = features.iloc[val_idx].astype(np.float32)
            
            # Fit scaler on training data only
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            if self.prediction_type == 'next_day':
                # Scale targets for regression
                y_train = target.iloc[train_idx].values.reshape(-1, 1).astype(np.float32)
                y_val = target.iloc[val_idx].values.reshape(-1, 1).astype(np.float32)
                
                # Fit target scaler on training data only
                y_train_scaled = self.target_scaler.fit_transform(y_train)
                y_val_scaled = self.target_scaler.transform(y_val)
                
                y_train = torch.FloatTensor(y_train_scaled)
                y_val = torch.FloatTensor(y_val_scaled)
            else:
                # For classification, convert targets to integers
                y_train = torch.LongTensor(target.iloc[train_idx].values.astype(np.int64))
                y_val = torch.LongTensor(target.iloc[val_idx].values.astype(np.int64))
            
            X_train = torch.FloatTensor(X_train_scaled)
            X_val = torch.FloatTensor(X_val_scaled)
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            return train_loader, val_loader
        except Exception as e:
            self.logger.error(f"Error in MLP data preparation: {str(e)}")
            raise  # Re-raise to prevent silent failures

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
                self.logger.info(f"Training fold {fold+1}/5")
                
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
                    
                    self.logger.info(f"Fold {fold+1}, Epoch [{epoch+1}/{self.epochs}], "
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
                            self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
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
            
            self.logger.info(f"Completed walk-forward validation for MLP")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in MLP training: {str(e)}")
            return None

    def train(self):
        """Train model with optional hyperparameter tuning."""
        try:
            # Load data for all model types
            features, target = self.load_data()
            if features is None or target is None:
                return None
            
            # Store data for later use
            self.features = features
            self.target = target
            
            # Perform hyperparameter tuning if enabled
            if self.tune_hyperparameters:
                self._tune_hyperparameters()
            
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
                self.logger.error(f"Unsupported model type: {self.model_type}")
                return None
            
            if results is None:
                return None
            
            # Add hyperparameters to results
            if self.best_params:
                results['hyperparameters'] = self.best_params
            
            # Save results
            self.save_results(results)
            
            # Save model if not already saved (LSTM and MLP save per fold)
            if self.model_type in ['ridge', 'lasso', 'rf', 'xgb']:
                self.save_model()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            return None

    def _tune_hyperparameters(self) -> None:
        """Perform hyperparameter tuning based on model type."""
        try:
            if self.model_type in ['ridge', 'lasso', 'rf', 'xgb']:
                self.logger.info(f"Starting hyperparameter tuning for {self.model_type}")
                
                # Time-series CV for traditional models
                tscv = TimeSeriesSplit(n_splits=5)
                _, self.best_params = HyperparameterTuner.tune_traditional_model(
                    self.model_type,
                    self.features.values,
                    self.target.values,
                    tscv
                )
                
                self.logger.info(f"Completed tuning for {self.model_type} with best parameters: {self.best_params}")
                
            elif self.model_type in ['mlp', 'lstm']:
                self.logger.info(f"Starting hyperparameter tuning for {self.model_type}")
                
                # Deep learning models
                tscv = TimeSeriesSplit(n_splits=5)
                for split_i, (train_idx, val_idx) in enumerate(tscv.split(self.features), start=1):
                    self.logger.info(f"Tuning on split {split_i}/5")
                    self.best_params = HyperparameterTuner.tune_deep_learning(
                        self.model_type,
                        self.features,
                        self.target,
                        train_idx,
                        val_idx
                    )
                    if self.best_params:
                        self.logger.info(f"Completed split {split_i}/5 with best parameters: {self.best_params}")
                    break  # Only use first split for tuning to save time
                
                # Update model parameters
                if self.best_params:
                    if self.model_type == 'mlp':
                        self.hidden_sizes = self.best_params['hidden_sizes']
                        self.dropout = self.best_params['dropout']
                        self.learning_rate = self.best_params['learning_rate']
                        self.batch_size = self.best_params['batch_size']
                    else:  # lstm
                        self.hidden_size = self.best_params['hidden_size']
                        self.num_layers = self.best_params['num_layers']
                        self.dropout = self.best_params['dropout']
                        self.learning_rate = self.best_params['learning_rate']
                        self.batch_size = self.best_params['batch_size']
                        self.sequence_length = self.best_params['sequence_length']
            
            elif self.model_type == 'arima':
                self.logger.info("Starting hyperparameter tuning for ARIMA")
                self.best_params, _ = HyperparameterTuner.tune_arima(self.target.values)
                self.logger.info(f"Completed ARIMA tuning with best parameters: {self.best_params}")
            
            self.logger.info(f"Best parameters found: {self.best_params}")
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {str(e)}")
            self.best_params = None

    @staticmethod
    def run_batch_training():
        """Run training for 2s10s spread with all prediction types and appropriate models."""
        # Get module-level logger
        logger = logging.getLogger("ModelTrainingBatch")
        
        # Define prediction types and their corresponding models
        prediction_models = {
            'next_day': ['arima', 'mlp', 'lstm', 'xgb', 'rf', 'lasso', 'ridge'],  
            'direction': ['mlp', 'lstm', 'xgb', 'rf'],
            'ternary': ['mlp', 'lstm', 'xgb', 'rf']
        }
        
        # Focus only on 2s10s spread
        spread = '2s10s'
        
        # Create results directory if it doesn't exist
        Path('results/model_training').mkdir(parents=True, exist_ok=True)
        
        # Initialize results DataFrame
        results_summary = []
        
        # Run training for each prediction type and its corresponding models
        for pred_type, model_types in prediction_models.items():
            for model_type in model_types:
                try:
                    logger.info(f"Training {spread} {pred_type} with {model_type}")
                    
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
                        
                        logger.info(f"Completed {spread} {pred_type} with {model_type}")
                        logger.info(f"Results: {summary}")
                    else:
                        logger.error(f"Training failed for {spread} {pred_type} with {model_type}")
                        
                except Exception as e:
                    logger.error(f"Error training {spread} {pred_type} with {model_type}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Save summary results
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_file = Path('results/model_training/training_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved training summary to {summary_file}")

if __name__ == "__main__":
    ModelTrainer.run_batch_training() 