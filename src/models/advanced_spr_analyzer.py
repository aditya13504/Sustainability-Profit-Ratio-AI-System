"""
Advanced Sustainability Profit Ratio (SPR) Analyzer

This module implements a comprehensive SPR analysis system that integrates:
- Predictive analytics using machine learning
- Qualitative data analysis with sentiment processing
- Industry-specific metrics and benchmarks
- Rigorous backtesting and validation
- Continuous model improvement
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import pipeline
import yfinance as yf
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config_loader import ConfigLoader
from utils.logging_utils import get_logger
from utils.sentiment_analyzer import SentimentAnalyzer
from models.industry_metrics import IndustryMetrics
from financial.data_processor import FinancialDataProcessor


class AdvancedSPRAnalyzer:
    """
    Advanced SPR Analyzer with predictive capabilities, sentiment analysis,
    and industry-specific metrics
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Advanced SPR Analyzer"""
        self.config = ConfigLoader(config_path).config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.industry_metrics = IndustryMetrics()
        self.financial_processor = FinancialDataProcessor()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Initialize ML models
        self._initialize_models()
        
        # Backtesting results
        self.backtest_results = {}
        
        self.logger.info("Advanced SPR Analyzer initialized successfully")
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    async def analyze_company_advanced(self, symbol: str, 
                                     include_predictions: bool = True,
                                     include_sentiment: bool = True,
                                     include_industry_analysis: bool = True,
                                     prediction_horizon: int = 12) -> Dict[str, Any]:
        """
        Perform comprehensive SPR analysis with advanced features
        
        Args:
            symbol: Stock symbol
            include_predictions: Include predictive analytics
            include_sentiment: Include sentiment analysis
            include_industry_analysis: Include industry-specific metrics
            prediction_horizon: Prediction horizon in months
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info(f"Starting advanced SPR analysis for {symbol}")
        
        try:
            # Base SPR analysis
            base_results = await self._get_base_spr_analysis(symbol)
            
            # Enhanced analysis components
            results = {
                'symbol': symbol.upper(),
                'timestamp': datetime.now().isoformat(),
                'base_spr': base_results,
                'advanced_features': {}
            }
            
            # Predictive analytics
            if include_predictions:
                predictions = await self._generate_predictions(symbol, prediction_horizon)
                results['advanced_features']['predictions'] = predictions
            
            # Sentiment analysis
            if include_sentiment:
                sentiment = await self._analyze_sentiment(symbol)
                results['advanced_features']['sentiment'] = sentiment
            
            # Industry analysis
            if include_industry_analysis:
                industry_analysis = await self._analyze_industry_metrics(symbol)
                results['advanced_features']['industry'] = industry_analysis
            
            # Calculate enhanced SPR score
            enhanced_spr = self._calculate_enhanced_spr(results)
            results['enhanced_spr_score'] = enhanced_spr
            
            # Risk assessment
            risk_metrics = await self._assess_risks(symbol, results)
            results['risk_assessment'] = risk_metrics
            
            # Recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            self.logger.info(f"Advanced SPR analysis completed for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in advanced SPR analysis for {symbol}: {str(e)}")
            raise
    
    async def _get_base_spr_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get base SPR analysis results"""
        # This would integrate with the existing SPR analyzer
        return {
            'spr_score': 7.5,  # Placeholder
            'profit_performance': 8.2,
            'sustainability_impact': 7.8,
            'research_alignment': 6.9
        }
    
    async def _generate_predictions(self, symbol: str, horizon: int) -> Dict[str, Any]:
        """Generate predictive analytics for the company"""
        self.logger.info(f"Generating predictions for {symbol} with {horizon} month horizon")
        
        try:
            # Collect historical data
            historical_data = await self._collect_historical_data(symbol)
            
            # Feature engineering
            features = self._engineer_features(historical_data)
            
            # Generate predictions for each model
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Train or load model
                    trained_model = await self._train_or_load_model(model_name, features)
                    
                    # Generate predictions
                    future_predictions = self._predict_future_performance(
                        trained_model, features, horizon
                    )
                    
                    predictions[model_name] = future_predictions
                    
                except Exception as e:
                    self.logger.error(f"Error in {model_name} predictions: {str(e)}")
                    predictions[model_name] = {'error': str(e)}
            
            # Ensemble predictions
            ensemble_prediction = self._create_ensemble_prediction(predictions)
            
            return {
                'individual_models': predictions,
                'ensemble': ensemble_prediction,
                'confidence_intervals': self._calculate_confidence_intervals(predictions),
                'feature_importance': self._get_feature_importance(),
                'prediction_horizon_months': horizon
            }
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from news and social media"""
        self.logger.info(f"Analyzing sentiment for {symbol}")
        
        try:
            # Get news articles and social media data
            news_data = await self._collect_news_data(symbol)
            social_data = await self._collect_social_media_data(symbol)
            
            # Analyze sentiment
            news_sentiment = self.sentiment_analyzer.analyze_batch(news_data)
            social_sentiment = self.sentiment_analyzer.analyze_batch(social_data)
            
            # Calculate aggregate sentiment scores
            sentiment_results = {
                'news_sentiment': {
                    'overall_score': np.mean([s['score'] for s in news_sentiment]),
                    'positive_ratio': len([s for s in news_sentiment if s['label'] == 'POSITIVE']) / len(news_sentiment),
                    'negative_ratio': len([s for s in news_sentiment if s['label'] == 'NEGATIVE']) / len(news_sentiment),
                    'neutral_ratio': len([s for s in news_sentiment if s['label'] == 'NEUTRAL']) / len(news_sentiment),
                    'article_count': len(news_sentiment)
                },
                'social_sentiment': {
                    'overall_score': np.mean([s['score'] for s in social_sentiment]) if social_sentiment else 0,
                    'positive_ratio': len([s for s in social_sentiment if s['label'] == 'POSITIVE']) / len(social_sentiment) if social_sentiment else 0,
                    'negative_ratio': len([s for s in social_sentiment if s['label'] == 'NEGATIVE']) / len(social_sentiment) if social_sentiment else 0,
                    'neutral_ratio': len([s for s in social_sentiment if s['label'] == 'NEUTRAL']) / len(social_sentiment) if social_sentiment else 0,
                    'post_count': len(social_sentiment)
                },
                'sustainability_keywords': self._extract_sustainability_keywords(news_data + social_data),
                'sentiment_trend': self._calculate_sentiment_trend(news_sentiment, social_sentiment)
            }
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_industry_metrics(self, symbol: str) -> Dict[str, Any]:
        """Analyze industry-specific metrics"""
        self.logger.info(f"Analyzing industry metrics for {symbol}")
        
        try:
            # Get company industry
            company_info = yf.Ticker(symbol).info
            industry = company_info.get('industry', 'Unknown')
            sector = company_info.get('sector', 'Unknown')
            
            # Get industry-specific metrics
            industry_analysis = await self.industry_metrics.analyze_company(symbol, industry, sector)
            
            # Benchmark against industry peers
            peer_comparison = await self._benchmark_against_peers(symbol, industry)
            
            # Calculate industry-adjusted scores
            industry_adjusted_scores = self._calculate_industry_adjusted_scores(
                symbol, industry_analysis, peer_comparison
            )
            
            return {
                'industry': industry,
                'sector': sector,
                'industry_metrics': industry_analysis,
                'peer_comparison': peer_comparison,
                'industry_adjusted_scores': industry_adjusted_scores,
                'industry_sustainability_standards': self._get_industry_standards(industry)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing industry metrics for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_enhanced_spr(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate enhanced SPR score incorporating all advanced features"""
        base_spr = results['base_spr']['spr_score']
        
        # Weight factors
        weights = {
            'base': 0.4,
            'predictions': 0.25,
            'sentiment': 0.20,
            'industry': 0.15
        }
        
        enhanced_score = base_spr * weights['base']
        
        # Add prediction component
        if 'predictions' in results['advanced_features']:
            pred_component = self._extract_prediction_component(
                results['advanced_features']['predictions']
            )
            enhanced_score += pred_component * weights['predictions']
        
        # Add sentiment component
        if 'sentiment' in results['advanced_features']:
            sentiment_component = self._extract_sentiment_component(
                results['advanced_features']['sentiment']
            )
            enhanced_score += sentiment_component * weights['sentiment']
        
        # Add industry component
        if 'industry' in results['advanced_features']:
            industry_component = self._extract_industry_component(
                results['advanced_features']['industry']
            )
            enhanced_score += industry_component * weights['industry']
        
        return {
            'enhanced_spr_score': enhanced_score,
            'components': {
                'base_spr': base_spr * weights['base'],
                'prediction_impact': pred_component * weights['predictions'] if 'predictions' in results['advanced_features'] else 0,
                'sentiment_impact': sentiment_component * weights['sentiment'] if 'sentiment' in results['advanced_features'] else 0,
                'industry_impact': industry_component * weights['industry'] if 'industry' in results['advanced_features'] else 0
            },
            'confidence_level': self._calculate_confidence_level(results)
        }
    
    async def backtest_model(self, symbol: str, start_date: str, end_date: str, 
                           prediction_horizon: int = 6) -> Dict[str, Any]:
        """
        Perform rigorous backtesting of the SPR prediction model
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            prediction_horizon: Prediction horizon in months
            
        Returns:
            Comprehensive backtesting results
        """
        self.logger.info(f"Starting backtesting for {symbol} from {start_date} to {end_date}")
        
        try:
            # Collect historical data
            historical_data = await self._collect_backtesting_data(symbol, start_date, end_date)
            
            # Time series split for backtesting
            tscv = TimeSeriesSplit(n_splits=5)
            
            backtest_results = {
                'symbol': symbol,
                'backtest_period': {'start': start_date, 'end': end_date},
                'prediction_horizon_months': prediction_horizon,
                'model_performance': {},
                'predictions_vs_actual': [],
                'performance_metrics': {},
                'error_analysis': {}
            }
            
            # Test each model
            for model_name, model in self.models.items():
                model_results = await self._backtest_single_model(
                    model_name, model, historical_data, tscv, prediction_horizon
                )
                backtest_results['model_performance'][model_name] = model_results
            
            # Calculate overall performance metrics
            backtest_results['performance_metrics'] = self._calculate_backtest_metrics(
                backtest_results['model_performance']
            )
            
            # Error analysis
            backtest_results['error_analysis'] = self._analyze_prediction_errors(
                backtest_results['model_performance']
            )
            
            # Store results for future reference
            self.backtest_results[symbol] = backtest_results
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting for {symbol}: {str(e)}")
            raise
    
    async def _backtest_single_model(self, model_name: str, model: Any, 
                                   historical_data: pd.DataFrame, tscv: TimeSeriesSplit,
                                   prediction_horizon: int) -> Dict[str, Any]:
        """Backtest a single model"""
        predictions = []
        actuals = []
        feature_importance_scores = []
        
        for train_index, test_index in tscv.split(historical_data):
            # Split data
            train_data = historical_data.iloc[train_index]
            test_data = historical_data.iloc[test_index]
            
            # Feature engineering
            X_train = self._engineer_features(train_data)
            y_train = self._extract_target_variable(train_data, prediction_horizon)
            
            X_test = self._engineer_features(test_data)
            y_test = self._extract_target_variable(test_data, prediction_horizon)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            predictions.extend(y_pred)
            actuals.extend(y_test)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_importance_scores.append(model.feature_importances_)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(mse),
            'feature_importance': np.mean(feature_importance_scores, axis=0) if feature_importance_scores else None
        }
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk"""
        if model_name in self.models:
            model_data = {
                'model': self.models[model_name],
                'scaler': self.scalers[model_name],
                'feature_importance': self.feature_importance.get(model_name, None)
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model from disk"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models[model_name] = model_data['model']
            self.scalers[model_name] = model_data['scaler']
            if 'feature_importance' in model_data:
                self.feature_importance[model_name] = model_data['feature_importance']
            self.logger.info(f"Model {model_name} loaded from {filepath}")
        else:
            self.logger.error(f"Model file not found: {filepath}")
    
    # Helper methods (simplified implementations)
    
    async def _collect_historical_data(self, symbol: str) -> pd.DataFrame:
        """Collect historical data for the company"""
        # Implementation would fetch comprehensive historical data
        return pd.DataFrame()  # Placeholder
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        # Implementation would create comprehensive feature set
        return pd.DataFrame()  # Placeholder
    
    async def _train_or_load_model(self, model_name: str, features: pd.DataFrame):
        """Train model or load from cache"""
        # Implementation would handle model training/loading
        return self.models[model_name]  # Placeholder
    
    def _predict_future_performance(self, model, features: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate future performance predictions"""
        # Implementation would generate actual predictions
        return {'predictions': [7.5, 7.8, 8.1], 'confidence': 0.85}  # Placeholder
    
    def _create_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models"""
        # Implementation would combine predictions from different models
        return {'ensemble_score': 7.7, 'confidence': 0.82}  # Placeholder
    
    def _calculate_confidence_intervals(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence intervals for predictions"""
        return {'lower_bound': 6.8, 'upper_bound': 8.6}  # Placeholder
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {'sustainability_score': 0.35, 'financial_metrics': 0.25, 'sentiment': 0.20, 'industry_factors': 0.20}  # Placeholder
    
    async def _collect_news_data(self, symbol: str) -> List[str]:
        """Collect news articles about the company"""
        # Implementation would fetch actual news data
        return [f"Sample news article about {symbol} sustainability initiatives"]  # Placeholder
    
    async def _collect_social_media_data(self, symbol: str) -> List[str]:
        """Collect social media posts about the company"""
        # Implementation would fetch social media data
        return [f"Sample social media post about {symbol}"]  # Placeholder
    
    def _extract_sustainability_keywords(self, text_data: List[str]) -> List[str]:
        """Extract sustainability-related keywords"""
        sustainability_keywords = ['renewable', 'carbon', 'green', 'sustainable', 'ESG', 'environment']
        return sustainability_keywords  # Placeholder
    
    def _calculate_sentiment_trend(self, news_sentiment: List[Dict], social_sentiment: List[Dict]) -> Dict[str, Any]:
        """Calculate sentiment trend over time"""
        return {'trend': 'positive', 'momentum': 0.65}  # Placeholder
    
    async def _benchmark_against_peers(self, symbol: str, industry: str) -> Dict[str, Any]:
        """Benchmark company against industry peers"""
        return {'peer_ranking': 3, 'total_peers': 10, 'percentile': 70}  # Placeholder
    
    def _calculate_industry_adjusted_scores(self, symbol: str, industry_analysis: Dict, peer_comparison: Dict) -> Dict[str, float]:
        """Calculate industry-adjusted SPR scores"""
        return {'adjusted_spr': 7.8, 'industry_premium': 0.3}  # Placeholder
    
    def _get_industry_standards(self, industry: str) -> Dict[str, Any]:
        """Get sustainability standards for the industry"""
        return {'carbon_intensity_target': 50, 'renewable_energy_target': 80}  # Placeholder
    
    def _extract_prediction_component(self, predictions: Dict[str, Any]) -> float:
        """Extract prediction component for enhanced SPR"""
        return 7.5  # Placeholder
    
    def _extract_sentiment_component(self, sentiment: Dict[str, Any]) -> float:
        """Extract sentiment component for enhanced SPR"""
        return 7.2  # Placeholder
    
    def _extract_industry_component(self, industry: Dict[str, Any]) -> float:
        """Extract industry component for enhanced SPR"""
        return 8.0  # Placeholder
    
    def _calculate_confidence_level(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence level"""
        return 0.78  # Placeholder
    
    async def _collect_backtesting_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect data for backtesting"""
        return pd.DataFrame()  # Placeholder
    
    def _extract_target_variable(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Extract target variable for prediction"""
        return pd.Series()  # Placeholder
    
    def _calculate_backtest_metrics(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall backtesting metrics"""
        return {'overall_accuracy': 0.75, 'sharpe_ratio': 1.2}  # Placeholder
    
    def _analyze_prediction_errors(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction errors"""
        return {'error_distribution': 'normal', 'bias': 0.02}  # Placeholder
    
    async def _assess_risks(self, symbol: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various risks"""
        return {'market_risk': 0.3, 'esg_risk': 0.2, 'regulatory_risk': 0.25}  # Placeholder
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        return [
            "Increase renewable energy investments",
            "Improve supply chain sustainability",
            "Enhance ESG reporting transparency"
        ]  # Placeholder
