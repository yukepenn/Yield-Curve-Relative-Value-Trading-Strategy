"""
Feature Analysis Module

This module performs comprehensive analysis of features to:
1. Compute feature importance
2. Analyze feature correlations
3. Identify redundant features
4. Optimize feature set
5. Generate analysis reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """Class for analyzing and optimizing features."""
    
    def __init__(self, data_dir: str = 'data/processed'):
        """
        Initialize FeatureAnalyzer.
        
        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = Path(data_dir)
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> None:
        """Load processed features and targets."""
        logger.info("Loading processed data...")
        
        # Load features
        self.features = pd.read_csv(self.data_dir / 'features.csv', index_col=0)
        self.features.index = pd.to_datetime(self.features.index)
        
        # Load targets
        self.targets = pd.read_csv(self.data_dir / 'targets.csv', index_col=0)
        self.targets.index = pd.to_datetime(self.targets.index)
        
        # Align data
        self._align_data()
        
        # Clean data
        self._clean_data()
        
        logger.info(f"Loaded {len(self.features)} samples with {len(self.features.columns)} features")
        logger.info(f"Loaded {len(self.targets)} samples with {len(self.targets.columns)} targets")
        
    def _align_data(self) -> None:
        """Align features and targets to have the same samples."""
        logger.info("Aligning features and targets...")
        
        # Get common dates
        common_dates = self.features.index.intersection(self.targets.index)
        
        # Filter both dataframes to common dates
        self.features = self.features.loc[common_dates]
        self.targets = self.targets.loc[common_dates]
        
        # Sort by date
        self.features = self.features.sort_index()
        self.targets = self.targets.sort_index()
        
        logger.info(f"Data aligned with {len(common_dates)} common samples")
        
    def _clean_data(self) -> None:
        """Clean data by handling missing and infinite values."""
        logger.info("Cleaning data...")
        
        # Replace infinite values with NaN
        self.features = self.features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        self.features = self.features.ffill()
        
        # Backward fill any remaining missing values
        self.features = self.features.bfill()
        
        # Replace any remaining NaN with 0
        self.features = self.features.fillna(0)
        
        # Do the same for targets
        self.targets = self.targets.replace([np.inf, -np.inf], np.nan)
        self.targets = self.targets.ffill()
        self.targets = self.targets.bfill()
        self.targets = self.targets.fillna(0)
        
        logger.info("Data cleaning complete")
        
    def compute_feature_importance(self, target_col: str, task: str = 'regression') -> pd.DataFrame:
        """
        Compute feature importance using Random Forest.
        
        Args:
            target_col: Target column name
            task: 'regression' or 'classification'
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Computing feature importance for {target_col} ({task})...")
        
        # Prepare data
        X = self.features.copy()
        y = self.targets[target_col].copy()
        
        # Remove rows with missing values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare target for classification
        if task == 'classification':
            y = self.label_encoder.fit_transform(y)
            
        # Fit model
        if task == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            mi_func = mutual_info_regression
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            mi_func = mutual_info_classif
            
        model.fit(X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'mutual_info': mi_func(X_scaled, y)
        }).sort_values('importance', ascending=False)
        
        return importance
        
    def analyze_feature_correlations(self, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """
        Analyze feature correlations and identify highly correlated features.
        
        Args:
            threshold: Correlation threshold for identifying redundant features
            
        Returns:
            Tuple of (correlation matrix, list of features to remove)
        """
        logger.info("Analyzing feature correlations...")
        
        # Compute correlation matrix
        corr_matrix = self.features.corr()
        
        # Identify highly correlated features
        redundant_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    redundant_features.add(colname)
                    
        return corr_matrix, list(redundant_features)
        
    def analyze_feature_distributions(self) -> pd.DataFrame:
        """
        Analyze feature distributions and statistics.
        
        Returns:
            DataFrame with feature statistics
        """
        logger.info("Analyzing feature distributions...")
        
        stats = pd.DataFrame({
            'mean': self.features.mean(),
            'std': self.features.std(),
            'min': self.features.min(),
            'max': self.features.max(),
            'skew': self.features.skew(),
            'kurtosis': self.features.kurtosis(),
            'missing_pct': self.features.isna().mean() * 100
        })
        
        return stats
        
    def analyze_feature_target_relationships(self, target_col: str) -> pd.DataFrame:
        """
        Analyze relationships between features and target.
        
        Args:
            target_col: Target column name
            
        Returns:
            DataFrame with feature-target relationships
        """
        logger.info(f"Analyzing feature-target relationships for {target_col}...")
        
        relationships = []
        for feature in self.features.columns:
            # Compute correlation
            corr = spearmanr(self.features[feature], self.targets[target_col])[0]
            
            # Compute mutual information
            mi = mutual_info_regression(
                self.features[[feature]], 
                self.targets[target_col]
            )[0]
            
            relationships.append({
                'feature': feature,
                'correlation': corr,
                'mutual_info': mi
            })
            
        return pd.DataFrame(relationships).sort_values('mutual_info', ascending=False)
        
    def optimize_feature_set(self, target_col: str, task: str = 'regression', 
                           importance_threshold: float = 0.001,
                           correlation_threshold: float = 0.95) -> List[str]:
        """
        Optimize feature set by removing unimportant and redundant features.
        
        Args:
            target_col: Target column name
            task: 'regression' or 'classification'
            importance_threshold: Minimum importance score to keep
            correlation_threshold: Maximum correlation allowed
            
        Returns:
            List of selected features
        """
        logger.info("Optimizing feature set...")
        
        # Get feature importance
        importance = self.compute_feature_importance(target_col, task)
        important_features = set(importance[importance['importance'] > importance_threshold]['feature'])
        
        # Get correlation analysis
        _, redundant_features = self.analyze_feature_correlations(correlation_threshold)
        
        # Select features
        selected_features = list(important_features - set(redundant_features))
        
        logger.info(f"Selected {len(selected_features)} features out of {len(self.features.columns)}")
        return selected_features
        
    def generate_analysis_report(self, output_dir: str = 'results/feature_analysis') -> None:
        """
        Generate comprehensive feature analysis report.
        
        Args:
            output_dir: Directory to save analysis results
        """
        logger.info("Generating feature analysis report...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics
        stats = self.analyze_feature_distributions()
        stats.to_csv(output_dir / 'feature_stats.csv')
        
        # Analyze each target
        for target_col in self.targets.columns:
            # Create target directory
            target_dir = output_dir / target_col
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine task type
            task = 'regression' if 'delta' in target_col else 'classification'
            
            # Compute feature importance
            importance = self.compute_feature_importance(target_col, task)
            importance.to_csv(target_dir / 'importance.csv')
            
            # Analyze correlations
            corr_matrix, redundant = self.analyze_feature_correlations()
            corr_matrix.to_csv(target_dir / 'correlations.csv')
            pd.Series(redundant).to_csv(target_dir / 'redundant_features.csv')
            
            # Analyze feature-target relationships
            relationships = self.analyze_feature_target_relationships(target_col)
            relationships.to_csv(target_dir / 'feature_target_relationships.csv')
            
            # Optimize feature set
            selected_features = self.optimize_feature_set(target_col, task)
            pd.Series(selected_features).to_csv(target_dir / 'selected_features.csv')
            
            # Save summary
            summary = {
                'total_features': len(self.features.columns),
                'redundant_features': len(redundant),
                'selected_features': len(selected_features),
                'top_10_features': importance['feature'].head(10).tolist(),
                'top_10_importance': importance['importance'].head(10).tolist()
            }
            pd.Series(summary).to_json(target_dir / 'summary.json')
            
        logger.info(f"Analysis report saved to {output_dir}")
        
def main():
    """Main function to run feature analysis."""
    analyzer = FeatureAnalyzer()
    analyzer.load_data()
    analyzer.generate_analysis_report()
    
if __name__ == '__main__':
    main() 