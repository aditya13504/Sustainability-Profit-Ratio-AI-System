"""
Predictive Models for Sustainability Profit Ratio Analysis

This module implements advanced machine learning models for predicting
the financial impact of sustainability initiatives with backtesting capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logging_utils import get_logger


@dataclass
class BacktestResult:
    """Results from model backtesting"""
    model_name: str
    predictions: List[float]
    actual_values: List[float]
    mse: float
    mae: float
    r2: float
    accuracy_score: float
    prediction_dates: List[datetime]
    feature_importance: Dict[str, float]


@dataclass
class PredictionResult:
    """Results from model prediction"""
    predicted_value: float
    confidence_interval: Tuple[float, float]
    feature_contributions: Dict[str, float]
    prediction_date: datetime
    model_used: str


class PredictiveModelsManager:
    """
    Manages predictive models for SPR analysis with backtesting capabilities
    """
    
    def __init__(self, config_path: str = None):
        """Initialize predictive models manager"""
        self.config = ConfigLoader(config_path).config
        self.logger = get_logger(__name__)
        
        # Initialize models
        self.models = self._initialize_models()
        self.scalers = {}
        self.feature_selectors = {}
        
        # Model performance tracking
        self.model_performance = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize machine learning models"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=self.config['models']['ml']['random_forest']['n_estimators'],
                max_depth=self.config['models']['ml']['random_forest']['max_depth'],
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=self.config['models']['ml']['gradient_boosting']['n_estimators'],
                learning_rate=self.config['models']['ml']['gradient_boosting']['learning_rate'],
                max_depth=self.config['models']['ml']['gradient_boosting']['max_depth'],
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=tuple(self.config['models']['ml']['neural_network']['hidden_layers']),
                max_iter=self.config['models']['ml']['neural_network']['max_iter'],
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        self.logger.info(f"Initialized {len(models)} predictive models")
        return models
    
    def prepare_features(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        
        Args:
            data: Raw data DataFrame
            target_column: Name of target variable column
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        # Separate features and target
        features = data.drop(columns=[target_column])
        target = data[target_column]
        
        # Handle missing values
        features = features.fillna(features.mean())
        target = target.fillna(target.mean())
        
        # Feature engineering
        features = self._engineer_features(features)
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        self.logger.info(f"Prepared {len(numeric_features.columns)} features for training")
        return numeric_features, target
    
    def _engineer_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data
        
        Args:
            features: Original features DataFrame
            
        Returns:
            Enhanced features DataFrame
        """
        engineered_features = features.copy()
        
        # Sustainability efficiency ratios
        if 'carbon_emission_reduction' in features.columns and 'revenue' in features.columns:
            engineered_features['carbon_efficiency'] = (
                features['carbon_emission_reduction'] / features['revenue'].replace(0, 1)
            )
        
        if 'energy_efficiency_score' in features.columns and 'total_assets' in features.columns:
            engineered_features['energy_asset_ratio'] = (
                features['energy_efficiency_score'] / features['total_assets'].replace(0, 1)
            )
        
        # ESG composite scores
        esg_columns = [col for col in features.columns if any(term in col.lower() 
                      for term in ['esg', 'environmental', 'social', 'governance'])]
        if len(esg_columns) > 1:
            engineered_features['esg_composite'] = features[esg_columns].mean(axis=1)
        
        # Financial health indicators
        if all(col in features.columns for col in ['current_ratio', 'debt_to_equity']):
            engineered_features['financial_health'] = (
                features['current_ratio'] / (1 + features['debt_to_equity'])
            )
        
        # Industry-specific features
        if 'industry' in features.columns:
            # Create industry dummy variables
            industry_dummies = pd.get_dummies(features['industry'], prefix='industry')
            engineered_features = pd.concat([engineered_features, industry_dummies], axis=1)
        
        return engineered_features
    
    def train_models(self, data: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all models on the provided data
        
        Args:
            data: Training data
            target_column: Name of target variable
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Starting model training process")
        
        # Prepare features
        X, y = self.prepare_features(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Feature selection
        selector = SelectKBest(
            f_regression, 
            k=min(len(X.columns), self.config['models']['ml']['max_features'])
        )
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        self.feature_selectors['main'] = selector
        
        # Train models
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name} model")
                
                # Train model
                if model_name == 'neural_network':
                    # Neural networks need scaled data
                    model.fit(X_train_selected, y_train)
                else:
                    # Tree-based models can use original features for interpretability
                    model.fit(X_train, y_train)
                
                # Make predictions
                if model_name == 'neural_network':
                    y_pred = model.predict(X_test_selected)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                if model_name == 'neural_network':
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Feature importance (for tree-based models)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    feature_names = X.columns
                    feature_importance = dict(zip(feature_names, importance_scores))
                
                training_results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': feature_importance,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                self.logger.info(f"{model_name} training complete - R²: {r2:.3f}, MSE: {mse:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        # Store training results
        self.model_performance = training_results
        
        return training_results
    
    def backtest_models(self, data: pd.DataFrame, target_column: str, 
                       lookback_periods: int = 12) -> Dict[str, BacktestResult]:
        """
        Perform backtesting on trained models
        
        Args:
            data: Historical data for backtesting
            target_column: Name of target variable
            lookback_periods: Number of periods to look back for training
            
        Returns:
            Dictionary of backtesting results
        """
        self.logger.info("Starting model backtesting")
        
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Prepare features
        X, y = self.prepare_features(data, target_column)
        
        backtest_results = {}
        
        # Time series split for backtesting
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Backtesting {model_name}")
                
                all_predictions = []
                all_actuals = []
                all_dates = []
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Scale features if needed
                    if model_name == 'neural_network':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Retrain model on this fold
                        model_copy = MLPRegressor(
                            hidden_layer_sizes=tuple(self.config['models']['ml']['neural_network']['hidden_layers']),
                            max_iter=self.config['models']['ml']['neural_network']['max_iter'],
                            random_state=42
                        )
                        model_copy.fit(X_train_scaled, y_train)
                        predictions = model_copy.predict(X_test_scaled)
                    else:
                        # Retrain model on this fold
                        model_copy = self.models[model_name].__class__(**self.models[model_name].get_params())
                        model_copy.fit(X_train, y_train)
                        predictions = model_copy.predict(X_test)
                    
                    all_predictions.extend(predictions)
                    all_actuals.extend(y_test.values)
                    
                    # Add dates if available
                    if 'date' in data.columns:
                        all_dates.extend(data.iloc[test_idx]['date'].values)
                    else:
                        all_dates.extend([datetime.now() + timedelta(days=i) for i in range(len(test_idx))])
                
                # Calculate overall metrics
                mse = mean_squared_error(all_actuals, all_predictions)
                mae = mean_absolute_error(all_actuals, all_predictions)
                r2 = r2_score(all_actuals, all_predictions)
                
                # Calculate accuracy score (percentage of predictions within 10% of actual)
                accuracy_score = np.mean(
                    np.abs(np.array(all_predictions) - np.array(all_actuals)) / 
                    np.abs(np.array(all_actuals)) <= 0.1
                ) * 100
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                
                backtest_results[model_name] = BacktestResult(
                    model_name=model_name,
                    predictions=all_predictions,
                    actual_values=all_actuals,
                    mse=mse,
                    mae=mae,
                    r2=r2,
                    accuracy_score=accuracy_score,
                    prediction_dates=all_dates,
                    feature_importance=feature_importance
                )
                
                self.logger.info(f"{model_name} backtest complete - R²: {r2:.3f}, Accuracy: {accuracy_score:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error backtesting {model_name}: {e}")
        
        return backtest_results
    
    def predict_future_performance(self, features: pd.DataFrame, 
                                 model_name: str = 'random_forest') -> PredictionResult:
        """
        Predict future financial performance based on sustainability metrics
        
        Args:
            features: Features for prediction
            model_name: Name of model to use for prediction
            
        Returns:
            Prediction result with confidence intervals
        """
        try:
            model = self.models[model_name]
            
            # Prepare features
            features_processed = self._engineer_features(features)
            numeric_features = features_processed.select_dtypes(include=[np.number])
            
            # Handle missing values
            numeric_features = numeric_features.fillna(numeric_features.mean())
            
            # Scale if needed
            if model_name == 'neural_network' and 'main' in self.scalers:
                # Ensure features match training features
                scaler = self.scalers['main']
                features_scaled = scaler.transform(numeric_features)
                
                if 'main' in self.feature_selectors:
                    features_selected = self.feature_selectors['main'].transform(features_scaled)
                    prediction = model.predict(features_selected)[0]
                else:
                    prediction = model.predict(features_scaled)[0]
            else:
                prediction = model.predict(numeric_features)[0]
            
            # Calculate confidence interval (simplified approach)
            # In practice, you might use prediction intervals from the model
            std_error = 0.1 * abs(prediction)  # Assume 10% standard error
            confidence_interval = (prediction - 1.96 * std_error, prediction + 1.96 * std_error)
            
            # Feature contributions (for tree-based models)
            feature_contributions = {}
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = numeric_features.columns
                for name, importance in zip(feature_names, importances):
                    feature_contributions[name] = importance * numeric_features[name].iloc[0]
            
            return PredictionResult(
                predicted_value=prediction,
                confidence_interval=confidence_interval,
                feature_contributions=feature_contributions,
                prediction_date=datetime.now(),
                model_used=model_name
            )
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def get_best_model(self) -> str:
        """
        Get the name of the best performing model based on cross-validation scores
        
        Returns:
            Name of best model
        """
        if not self.model_performance:
            return 'random_forest'  # Default
        
        best_model = max(
            self.model_performance.keys(),
            key=lambda x: self.model_performance[x].get('cv_mean', 0)
        )
        
        return best_model
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'performance': self.model_performance
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_selectors = model_data['feature_selectors']
            self.model_performance = model_data['performance']
            self.logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


# Example usage
if __name__ == "__main__":
    # Example of using the predictive models
    import pandas as pd
    
    # Sample data
    sample_data = pd.DataFrame({
        'carbon_emission_reduction': [10, 20, 15, 25, 30],
        'energy_efficiency_score': [0.7, 0.8, 0.75, 0.85, 0.9],
        'water_usage_reduction': [5, 10, 8, 12, 15],
        'esg_score': [70, 80, 75, 85, 90],
        'revenue': [1000000, 1200000, 1100000, 1300000, 1400000],
        'future_profit_margin': [0.12, 0.15, 0.13, 0.16, 0.18]  # Target variable
    })
    
    # Initialize and train models
    predictor = PredictiveModelsManager()
    
    # Train models
    training_results = predictor.train_models(sample_data, 'future_profit_margin')
    print("Training Results:", training_results)
    
    # Make a prediction
    new_features = pd.DataFrame({
        'carbon_emission_reduction': [20],
        'energy_efficiency_score': [0.82],
        'water_usage_reduction': [10],
        'esg_score': [78],
        'revenue': [1250000]
    })
    
    prediction = predictor.predict_future_performance(new_features)
    print(f"Predicted profit margin: {prediction.predicted_value:.3f}")
    print(f"Confidence interval: {prediction.confidence_interval}")
