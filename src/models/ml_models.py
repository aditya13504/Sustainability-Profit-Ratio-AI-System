"""
Machine Learning Models for the SPR Analyzer

This module contains ML models for predicting SPR scores, classifying
sustainability practices, and forecasting financial performance.
"""

import logging
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader


@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    model_name: str
    rmse: float
    mae: float
    r2_score: float
    cross_val_score: float
    feature_importance: Dict[str, float] = None


class SPRPredictor:
    """
    Machine Learning model to predict SPR scores based on financial and sustainability metrics
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the SPR predictor"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
        }
        
        self.scalers = {}
        self.trained_models = {}
        self.model_performance = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def prepare_features(self, financial_data: List[Dict], sustainability_data: List[Dict]) -> pd.DataFrame:
        """
        Prepare feature matrix from financial and sustainability data
        
        Args:
            financial_data: List of financial metrics dictionaries
            sustainability_data: List of sustainability metrics dictionaries
            
        Returns:
            DataFrame with prepared features
        """
        try:
            # Convert to DataFrames
            financial_df = pd.DataFrame(financial_data)
            sustainability_df = pd.DataFrame(sustainability_data)
            
            # Merge on symbol
            combined_df = pd.merge(financial_df, sustainability_df, on='symbol', how='inner')
            
            # Select and engineer features
            feature_columns = [
                'revenue', 'net_income', 'total_assets', 'market_cap',
                'roi', 'profit_margin', 'debt_to_equity',
                'esg_score', 'environmental_score', 'social_score', 'governance_score'
            ]
            
            # Handle missing values
            for col in feature_columns:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(combined_df[col].median())
                else:
                    combined_df[col] = 0
                    
            # Feature engineering
            combined_df['revenue_per_asset'] = combined_df['revenue'] / (combined_df['total_assets'] + 1)
            combined_df['esg_financial_ratio'] = combined_df['esg_score'] * combined_df['roi']
            combined_df['sustainability_efficiency'] = (
                combined_df['environmental_score'] + combined_df['social_score']
            ) / 2
            
            feature_columns.extend(['revenue_per_asset', 'esg_financial_ratio', 'sustainability_efficiency'])
            
            return combined_df[['symbol'] + feature_columns]
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
            
    def train_models(self, features: pd.DataFrame, targets: List[float]) -> Dict[str, ModelPerformance]:
        """
        Train SPR prediction models
        
        Args:
            features: Feature matrix
            targets: Target SPR scores
            
        Returns:
            Dictionary of model performance metrics
        """
        try:
            if features.empty or not targets:
                raise ValueError("Invalid training data")
                
            # Prepare data
            X = features.drop('symbol', axis=1)
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['spr_predictor'] = scaler
            
            performance_results = {}
            
            # Train each model
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name} model...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                cv_score = cv_scores.mean()
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    
                performance = ModelPerformance(
                    model_name=model_name,
                    rmse=rmse,
                    mae=mae,
                    r2_score=r2,
                    cross_val_score=cv_score,
                    feature_importance=feature_importance
                )
                
                performance_results[model_name] = performance
                self.trained_models[model_name] = model
                self.model_performance[model_name] = performance
                
                self.logger.info(f"{model_name} - RMSE: {rmse:.3f}, R2: {r2:.3f}")
                
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
            
    def predict_spr(self, features: pd.DataFrame, model_name: str = 'random_forest') -> List[float]:
        """
        Predict SPR scores using trained model
        
        Args:
            features: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            List of predicted SPR scores
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
                
            # Prepare features
            X = features.drop('symbol', axis=1)
            X_scaled = self.scalers['spr_predictor'].transform(X)
            
            # Make predictions
            predictions = self.trained_models[model_name].predict(X_scaled)
            
            # Ensure predictions are within valid range (0-10)
            predictions = np.clip(predictions, 0, 10)
            
            return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return []
            
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.trained_models,
                'scalers': self.scalers,
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
            self.trained_models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_performance = model_data['performance']
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


class SustainabilityClassifier:
    """
    Text classifier for sustainability practices and their impact categories
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the sustainability classifier"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
        # Sustainability categories
        self.sustainability_categories = [
            'renewable_energy',
            'carbon_reduction',
            'waste_management',
            'water_conservation',
            'sustainable_supply_chain',
            'biodiversity',
            'social_responsibility',
            'governance'
        ]
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def prepare_training_data(self, research_insights: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Prepare training data from research insights
        
        Args:
            research_insights: List of research insight dictionaries
            
        Returns:
            Tuple of (texts, categories)
        """
        texts = []
        categories = []
        
        for insight in research_insights:
            if 'impact_description' in insight and 'practice' in insight:
                texts.append(insight['impact_description'])
                categories.append(insight['practice'].lower().replace(' ', '_'))
                
        return texts, categories
        
    def train_classifier(self, texts: List[str], categories: List[str]):
        """
        Train the sustainability classifier
        
        Args:
            texts: List of text descriptions
            categories: List of corresponding categories
        """
        try:
            if not texts or not categories:
                raise ValueError("Invalid training data")
                
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Encode categories
            y = self.label_encoder.fit_transform(categories)
            
            # Train classifier
            self.classifier.fit(X, y)
            
            self.logger.info("Sustainability classifier trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {e}")
            
    def classify_sustainability_text(self, text: str) -> Dict[str, float]:
        """
        Classify sustainability text into categories
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary of category probabilities
        """
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X)[0]
                categories = self.label_encoder.classes_
                
                return dict(zip(categories, probabilities))
            else:
                # For regression-based classifiers
                prediction = self.classifier.predict(X)[0]
                category = self.label_encoder.inverse_transform([int(prediction)])[0]
                
                return {category: 1.0}
                
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}")
            return {}


class FinancialForecaster:
    """
    Model to forecast financial performance based on sustainability initiatives
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the financial forecaster"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Time series model (simplified - in production, use more sophisticated models)
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def prepare_time_series_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for time series forecasting
        
        Args:
            historical_data: DataFrame with historical financial and sustainability data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Sort by date
            historical_data = historical_data.sort_values('date')
            
            # Create lag features
            for lag in [1, 3, 6, 12]:  # 1, 3, 6, 12 periods ago
                historical_data[f'revenue_lag_{lag}'] = historical_data['revenue'].shift(lag)
                historical_data[f'esg_score_lag_{lag}'] = historical_data['esg_score'].shift(lag)
                
            # Create moving averages
            for window in [3, 6, 12]:
                historical_data[f'revenue_ma_{window}'] = historical_data['revenue'].rolling(window).mean()
                historical_data[f'esg_ma_{window}'] = historical_data['esg_score'].rolling(window).mean()
                
            # Create trend features
            historical_data['revenue_trend'] = historical_data['revenue'].pct_change(periods=4)
            historical_data['esg_trend'] = historical_data['esg_score'].diff(periods=4)
            
            # Drop rows with NaN values
            historical_data = historical_data.dropna()
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error preparing time series features: {e}")
            return pd.DataFrame()
            
    def forecast_financial_impact(self, 
                                 current_metrics: Dict[str, float],
                                 sustainability_changes: Dict[str, float],
                                 periods: int = 12) -> Dict[str, List[float]]:
        """
        Forecast financial impact of sustainability changes
        
        Args:
            current_metrics: Current financial metrics
            sustainability_changes: Planned sustainability improvements
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasted metrics
        """
        try:
            # This is a simplified implementation
            # In production, you would use more sophisticated forecasting models
            
            forecasts = {
                'revenue': [],
                'profit_margin': [],
                'spr_score': []
            }
            
            # Base values
            base_revenue = current_metrics.get('revenue', 0)
            base_margin = current_metrics.get('profit_margin', 0)
            base_esg = current_metrics.get('esg_score', 5)
            
            # Sustainability impact factors (based on research correlations)
            esg_improvement = sustainability_changes.get('esg_score', 0)
            impact_factor = 1 + (esg_improvement * 0.02)  # 2% improvement per ESG point
            
            for period in range(periods):
                # Simple growth model with sustainability impact
                period_factor = 1 + (period * 0.01)  # 1% base growth per period
                
                forecasted_revenue = base_revenue * period_factor * impact_factor
                forecasted_margin = base_margin * (1 + esg_improvement * 0.005)  # Margin improvement
                forecasted_spr = (forecasted_margin / 10) * (base_esg + esg_improvement)
                
                forecasts['revenue'].append(forecasted_revenue)
                forecasts['profit_margin'].append(forecasted_margin)
                forecasts['spr_score'].append(forecasted_spr)
                
            return forecasts
            
        except Exception as e:
            self.logger.error(f"Error forecasting financial impact: {e}")
            return {}


class ModelEnsemble:
    """
    Ensemble of multiple models for improved SPR prediction accuracy
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model ensemble"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize component models
        self.spr_predictor = SPRPredictor(config_path)
        self.sustainability_classifier = SustainabilityClassifier(config_path)
        self.financial_forecaster = FinancialForecaster(config_path)
        
        # Ensemble weights
        self.ensemble_weights = {
            'spr_predictor': 0.5,
            'financial_forecaster': 0.3,
            'sustainability_classifier': 0.2
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def predict_ensemble_spr(self, 
                           financial_data: Dict[str, float],
                           sustainability_data: Dict[str, float],
                           research_text: str = "") -> float:
        """
        Make ensemble prediction for SPR score
        
        Args:
            financial_data: Financial metrics
            sustainability_data: Sustainability metrics
            research_text: Research-based insights text
            
        Returns:
            Ensemble SPR prediction
        """
        try:
            predictions = []
            weights = []
            
            # SPR Predictor
            if self.spr_predictor.trained_models:
                features_df = pd.DataFrame([{**financial_data, **sustainability_data}])
                spr_pred = self.spr_predictor.predict_spr(features_df)[0]
                predictions.append(spr_pred)
                weights.append(self.ensemble_weights['spr_predictor'])
                
            # Add other model predictions here as they become available
            
            # Calculate weighted ensemble prediction
            if predictions and weights:
                ensemble_prediction = np.average(predictions, weights=weights)
                return float(ensemble_prediction)
            else:
                # Fallback to simple calculation
                profit_score = financial_data.get('profit_margin', 0) / 2
                sustainability_score = sustainability_data.get('esg_score', 5)
                return (profit_score + sustainability_score) / 2
                
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 5.0  # Default neutral score


# Example usage and testing
if __name__ == "__main__":
    # Initialize models
    predictor = SPRPredictor()
    
    # Example training data preparation
    financial_data = [
        {
            'symbol': 'TSLA',
            'revenue': 81462000000,
            'net_income': 12556000000,
            'total_assets': 106618000000,
            'market_cap': 800000000000,
            'roi': 11.78,
            'profit_margin': 15.42,
            'debt_to_equity': 0.17
        }
    ]
    
    sustainability_data = [
        {
            'symbol': 'TSLA',
            'esg_score': 8.5,
            'environmental_score': 9.2,
            'social_score': 7.8,
            'governance_score': 8.5
        }
    ]
    
    target_spr_scores = [8.42]
    
    # Prepare features
    features = predictor.prepare_features(financial_data, sustainability_data)
    print("Features prepared:", features.shape)
    
    # Train models (would need more data in practice)
    print("Training models...")
    performance = predictor.train_models(features, target_spr_scores)
    
    for model_name, perf in performance.items():
        print(f"{model_name}: R2 = {perf.r2_score:.3f}, RMSE = {perf.rmse:.3f}")
        
    print("Model training example completed!")
