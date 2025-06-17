"""
Financial data processing module for the SPR Analyzer

This module handles the collection, processing, and analysis of financial data
from various sources to calculate profitability metrics.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader


@dataclass
class FinancialMetrics:
    """Data class for financial metrics"""
    symbol: str
    company_name: str
    market_cap: float
    revenue: float
    net_income: float
    total_assets: float
    total_debt: float
    cash_and_equivalents: float
    
    # Calculated ratios
    roi: float = 0.0
    profit_margin: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    revenue_growth: float = 0.0
    ebitda_margin: float = 0.0
    
    # Performance scores
    profitability_score: float = 0.0
    efficiency_score: float = 0.0
    growth_score: float = 0.0
    
    last_updated: datetime = None


@dataclass
class SustainabilityMetrics:
    """Data class for sustainability metrics"""
    symbol: str
    esg_score: float = 0.0
    environmental_score: float = 0.0
    social_score: float = 0.0
    governance_score: float = 0.0
    
    # Specific sustainability indicators
    carbon_intensity: float = 0.0
    renewable_energy_percentage: float = 0.0
    waste_reduction_percentage: float = 0.0
    water_usage_efficiency: float = 0.0
    
    # External ratings
    msci_rating: str = ""
    sustainalytics_score: float = 0.0
    
    last_updated: datetime = None


class FinancialDataProcessor:
    """
    Processes financial data from multiple sources to calculate profitability metrics
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the financial data processor"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize API clients
        self._setup_api_clients()
        
        # Cache for financial data
        self.data_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_api_clients(self):
        """Set up API clients for financial data sources"""
        try:
            # Alpha Vantage
            alpha_vantage_key = self.config['apis']['financial']['alpha_vantage']['api_key']
            if alpha_vantage_key and alpha_vantage_key != "${ALPHA_VANTAGE_API_KEY}":
                self.alpha_vantage_ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
                self.alpha_vantage_fd = FundamentalData(key=alpha_vantage_key, output_format='pandas')
            else:
                self.alpha_vantage_ts = None
                self.alpha_vantage_fd = None
                self.logger.warning("Alpha Vantage API key not configured")
                
        except Exception as e:
            self.logger.error(f"Error setting up API clients: {e}")
            
    async def get_financial_metrics(self, symbol: str) -> Optional[FinancialMetrics]:
        """
        Get comprehensive financial metrics for a company
        
        Args:
            symbol: Stock symbol (e.g., 'TSLA', 'GOOGL')
            
        Returns:
            FinancialMetrics object or None if data unavailable
        """
        try:
            self.logger.info(f"Fetching financial metrics for {symbol}")
            
            # Check cache first
            cache_key = f"financial_{symbol}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp']):
                    return cached_data['data']
                    
            # Fetch data from multiple sources
            yfinance_data = await self._get_yfinance_data(symbol)
            alpha_vantage_data = await self._get_alpha_vantage_data(symbol)
            
            # Combine and process data
            metrics = self._process_financial_data(symbol, yfinance_data, alpha_vantage_data)
            
            # Cache the result
            self.data_cache[cache_key] = {
                'data': metrics,
                'timestamp': datetime.now()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return None
            
    async def _get_yfinance_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            return {
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow
            }
            
        except Exception as e:
            self.logger.warning(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return {}
            
    async def _get_alpha_vantage_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage"""
        try:
            if not self.alpha_vantage_fd:
                return {}
                
            # Get company overview
            overview, _ = self.alpha_vantage_fd.get_company_overview(symbol)
            
            # Get income statement
            income_statement, _ = self.alpha_vantage_fd.get_income_statement_annual(symbol)
            
            # Get balance sheet
            balance_sheet, _ = self.alpha_vantage_fd.get_balance_sheet_annual(symbol)
            
            return {
                'overview': overview,
                'income_statement': income_statement,
                'balance_sheet': balance_sheet
            }
            
        except Exception as e:
            self.logger.warning(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return {}
            
    def _process_financial_data(self, symbol: str, yf_data: Dict, av_data: Dict) -> FinancialMetrics:
        """Process and combine financial data from multiple sources"""
        # Start with basic structure
        metrics = FinancialMetrics(
            symbol=symbol,
            company_name="Unknown",
            market_cap=0.0,
            revenue=0.0,
            net_income=0.0,
            total_assets=0.0,
            total_debt=0.0,
            cash_and_equivalents=0.0,
            last_updated=datetime.now()
        )
        
        # Process Yahoo Finance data
        if 'info' in yf_data:
            info = yf_data['info']
            metrics.company_name = info.get('longName', symbol)
            metrics.market_cap = info.get('marketCap', 0) or 0
            
        # Process financial statements
        if 'financials' in yf_data and not yf_data['financials'].empty:
            financials = yf_data['financials']
            if 'Total Revenue' in financials.index:
                metrics.revenue = financials.loc['Total Revenue'].iloc[0] if len(financials.columns) > 0 else 0
            if 'Net Income' in financials.index:
                metrics.net_income = financials.loc['Net Income'].iloc[0] if len(financials.columns) > 0 else 0
                
        if 'balance_sheet' in yf_data and not yf_data['balance_sheet'].empty:
            balance_sheet = yf_data['balance_sheet']
            if 'Total Assets' in balance_sheet.index:
                metrics.total_assets = balance_sheet.loc['Total Assets'].iloc[0] if len(balance_sheet.columns) > 0 else 0
            if 'Total Debt' in balance_sheet.index:
                metrics.total_debt = balance_sheet.loc['Total Debt'].iloc[0] if len(balance_sheet.columns) > 0 else 0
            if 'Cash And Cash Equivalents' in balance_sheet.index:
                metrics.cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if len(balance_sheet.columns) > 0 else 0
                
        # Calculate financial ratios
        metrics = self._calculate_financial_ratios(metrics)
        
        # Calculate performance scores
        metrics = self._calculate_performance_scores(metrics)
        
        return metrics
        
    def _calculate_financial_ratios(self, metrics: FinancialMetrics) -> FinancialMetrics:
        """Calculate key financial ratios"""
        try:
            # ROI (Return on Investment)
            if metrics.total_assets > 0:
                metrics.roi = (metrics.net_income / metrics.total_assets) * 100
                
            # Profit Margin
            if metrics.revenue > 0:
                metrics.profit_margin = (metrics.net_income / metrics.revenue) * 100
                
            # Debt to Equity (simplified)
            equity = metrics.total_assets - metrics.total_debt
            if equity > 0:
                metrics.debt_to_equity = metrics.total_debt / equity
                
            # Current Ratio (simplified estimation)
            if metrics.total_debt > 0:
                metrics.current_ratio = metrics.cash_and_equivalents / (metrics.total_debt * 0.3)  # Approximate current liabilities
                
        except Exception as e:
            self.logger.warning(f"Error calculating financial ratios: {e}")
            
        return metrics
        
    def _calculate_performance_scores(self, metrics: FinancialMetrics) -> FinancialMetrics:
        """Calculate normalized performance scores (0-10 scale)"""
        try:
            # Profitability Score (based on profit margin and ROI)
            profit_score = max(0, min(10, (metrics.profit_margin + 10) / 2))  # Normalize around 0% margin
            roi_score = max(0, min(10, (metrics.roi + 5) / 1.5))  # Normalize around 0% ROI
            metrics.profitability_score = (profit_score + roi_score) / 2
            
            # Efficiency Score (based on asset utilization)
            if metrics.total_assets > 0:
                asset_turnover = metrics.revenue / metrics.total_assets if metrics.total_assets > 0 else 0
                metrics.efficiency_score = max(0, min(10, asset_turnover * 5))
            else:
                metrics.efficiency_score = 0
                
            # Growth Score (placeholder - would need historical data for real calculation)
            metrics.growth_score = 5.0  # Default neutral score
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance scores: {e}")
            
        return metrics
        
    async def get_sustainability_metrics(self, symbol: str) -> Optional[SustainabilityMetrics]:
        """
        Get sustainability metrics for a company
        
        Args:
            symbol: Stock symbol
            
        Returns:
            SustainabilityMetrics object or None if data unavailable
        """
        try:
            self.logger.info(f"Fetching sustainability metrics for {symbol}")
            
            # Check cache first
            cache_key = f"sustainability_{symbol}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp']):
                    return cached_data['data']
                    
            # Fetch sustainability data
            sustainability_data = await self._get_sustainability_data(symbol)
            
            # Process data
            metrics = self._process_sustainability_data(symbol, sustainability_data)
            
            # Cache the result
            self.data_cache[cache_key] = {
                'data': metrics,
                'timestamp': datetime.now()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error fetching sustainability metrics for {symbol}: {e}")
            return None
            
    async def _get_sustainability_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch sustainability data from various sources"""
        try:
            # Try Yahoo Finance ESG data
            ticker = yf.Ticker(symbol)
            sustainability = getattr(ticker, 'sustainability', None)
            
            # For now, we'll use placeholder data since real ESG APIs require subscriptions
            # In production, you would integrate with actual ESG data providers
            
            return {
                'yfinance_sustainability': sustainability,
                'placeholder_data': self._generate_placeholder_sustainability_data(symbol)
            }
            
        except Exception as e:
            self.logger.warning(f"Error fetching sustainability data for {symbol}: {e}")
            return {}
            
    def _generate_placeholder_sustainability_data(self, symbol: str) -> Dict[str, Any]:
        """Generate placeholder sustainability data for demonstration"""
        # This is for demonstration purposes only
        # In production, replace with real ESG data sources
        
        base_scores = {
            'TSLA': {'esg': 8.5, 'env': 9.2, 'social': 7.8, 'gov': 8.5},
            'GOOGL': {'esg': 7.8, 'env': 8.1, 'social': 8.2, 'gov': 7.1},
            'MSFT': {'esg': 8.2, 'env': 7.9, 'social': 8.5, 'gov': 8.2},
            'AAPL': {'esg': 7.9, 'env': 8.0, 'social': 7.8, 'gov': 8.0}
        }
        
        if symbol in base_scores:
            return base_scores[symbol]
        else:
            # Generate random but realistic scores for other symbols
            import random
            return {
                'esg': round(random.uniform(5.0, 8.5), 1),
                'env': round(random.uniform(5.0, 9.0), 1),
                'social': round(random.uniform(5.0, 8.5), 1),
                'gov': round(random.uniform(5.0, 8.5), 1)
            }
            
    def _process_sustainability_data(self, symbol: str, data: Dict[str, Any]) -> SustainabilityMetrics:
        """Process sustainability data into metrics"""
        metrics = SustainabilityMetrics(
            symbol=symbol,
            last_updated=datetime.now()
        )
        
        # Process placeholder data (replace with real data processing in production)
        if 'placeholder_data' in data:
            placeholder = data['placeholder_data']
            metrics.esg_score = placeholder.get('esg', 0.0)
            metrics.environmental_score = placeholder.get('env', 0.0)
            metrics.social_score = placeholder.get('social', 0.0)
            metrics.governance_score = placeholder.get('gov', 0.0)
            
        # Process Yahoo Finance sustainability data if available
        if 'yfinance_sustainability' in data and data['yfinance_sustainability'] is not None:
            yf_sustainability = data['yfinance_sustainability']
            # Process actual Yahoo Finance sustainability data here
            # This would involve parsing the actual data structure
            
        return metrics
        
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid"""
        cache_duration = timedelta(seconds=self.config['data']['refresh_intervals']['financial_data'])
        return datetime.now() - timestamp < cache_duration
        
    async def get_industry_comparison(self, symbol: str, industry_symbols: List[str]) -> Dict[str, Any]:
        """
        Compare a company's financial metrics with industry peers
        
        Args:
            symbol: Target company symbol
            industry_symbols: List of industry peer symbols
            
        Returns:
            Dictionary with comparison data
        """
        try:
            self.logger.info(f"Performing industry comparison for {symbol}")
            
            # Get metrics for target company
            target_metrics = await self.get_financial_metrics(symbol)
            if not target_metrics:
                return {}
                
            # Get metrics for industry peers
            peer_metrics = []
            for peer_symbol in industry_symbols:
                if peer_symbol != symbol:
                    peer_metric = await self.get_financial_metrics(peer_symbol)
                    if peer_metric:
                        peer_metrics.append(peer_metric)
                        
            if not peer_metrics:
                return {"error": "No peer data available"}
                
            # Calculate industry averages
            industry_avg = self._calculate_industry_averages(peer_metrics)
            
            # Calculate percentile rankings
            rankings = self._calculate_percentile_rankings(target_metrics, peer_metrics)
            
            return {
                'target_company': target_metrics,
                'industry_average': industry_avg,
                'percentile_rankings': rankings,
                'peer_count': len(peer_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error in industry comparison: {e}")
            return {}
            
    def _calculate_industry_averages(self, peer_metrics: List[FinancialMetrics]) -> Dict[str, float]:
        """Calculate industry average metrics"""
        if not peer_metrics:
            return {}
            
        return {
            'roi': np.mean([m.roi for m in peer_metrics]),
            'profit_margin': np.mean([m.profit_margin for m in peer_metrics]),
            'debt_to_equity': np.mean([m.debt_to_equity for m in peer_metrics]),
            'profitability_score': np.mean([m.profitability_score for m in peer_metrics]),
            'efficiency_score': np.mean([m.efficiency_score for m in peer_metrics])
        }
        
    def _calculate_percentile_rankings(self, target: FinancialMetrics, peers: List[FinancialMetrics]) -> Dict[str, float]:
        """Calculate percentile rankings for target company vs peers"""
        rankings = {}
        
        metrics_to_rank = ['roi', 'profit_margin', 'profitability_score', 'efficiency_score']
        
        for metric in metrics_to_rank:
            target_value = getattr(target, metric)
            peer_values = [getattr(peer, metric) for peer in peers]
            
            # Calculate percentile
            percentile = (sum(1 for v in peer_values if v < target_value) / len(peer_values)) * 100
            rankings[metric] = percentile
            
        return rankings


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize processor
        processor = FinancialDataProcessor()
        
        # Test symbols
        symbols = ["TSLA", "GOOGL", "MSFT"]
        
        for symbol in symbols:
            print(f"\n=== {symbol} Financial Metrics ===")
            
            # Get financial metrics
            metrics = await processor.get_financial_metrics(symbol)
            if metrics:
                print(f"Company: {metrics.company_name}")
                print(f"Revenue: ${metrics.revenue:,.0f}")
                print(f"Net Income: ${metrics.net_income:,.0f}")
                print(f"ROI: {metrics.roi:.2f}%")
                print(f"Profit Margin: {metrics.profit_margin:.2f}%")
                print(f"Profitability Score: {metrics.profitability_score:.1f}/10")
                
            # Get sustainability metrics
            sustainability = await processor.get_sustainability_metrics(symbol)
            if sustainability:
                print(f"ESG Score: {sustainability.esg_score:.1f}/10")
                print(f"Environmental Score: {sustainability.environmental_score:.1f}/10")
                
    # Run example
    asyncio.run(main())
