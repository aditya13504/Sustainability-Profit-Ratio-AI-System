import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import json
import numpy as np
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import finnhub
import logging

# Comprehensive list of companies for maximum dataset coverage
S_P_500_COMPANIES = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH',
    'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'PFE', 'ABBV', 'BAC',
    'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'VZ',
    'ADBE', 'CMCSA', 'NFLX', 'NKE', 'CRM', 'NEE', 'MRK', 'T', 'ACN', 'BMY',
    'TXN', 'CVS', 'RTX', 'QCOM', 'LLY', 'HON', 'PM', 'UPS', 'AMD', 'AMGN',
    'WFC', 'IBM', 'SPGI', 'LOW', 'LMT', 'C', 'CAT', 'GS', 'MDLZ', 'INTU',
    'UNP', 'AXP', 'ISRG', 'BLK', 'DE', 'NOW', 'BKNG', 'SYK', 'ANTM', 'PLD',
    'TJX', 'GILD', 'CB', 'MO', 'ZTS', 'ADP', 'MMM', 'CI', 'USB', 'TGT',
    'CSX', 'CCI', 'REGN', 'SHW', 'MU', 'DUK', 'EOG', 'BSX', 'CL', 'SO',
    'PYPL', 'DXCM', 'MCO', 'PNC', 'EL', 'AON', 'ICE', 'LRCX', 'COP', 'EQIX',
    'SPG', 'ITW', 'CME', 'F', 'GM', 'EMR', 'FCX', 'PSA', 'D', 'PGR',
    'WELL', 'MSI', 'DG', 'ECL', 'EXC', 'NSC', 'SRE', 'AEP', 'GIS', 'ORLY',
    'HUM', 'KLAC', 'APD', 'CTAS', 'AFL', 'EW', 'MCK', 'ADI', 'CDNS', 'A',
    'HCA', 'SNPS', 'MNST', 'KMB', 'IDXX', 'MCHP', 'PAYX', 'MSCI', 'CMG', 'AZO',
    'PRU', 'ALL', 'TRV', 'AIG', 'TT', 'FAST', 'EA', 'CTSH', 'ADM', 'YUM',
    'ROST', 'OTIS', 'CARR', 'PCAR', 'KR', 'VRSK', 'VRTX', 'NXPI', 'SBUX', 'WM',
    'CHTR', 'CSGP', 'ANSS', 'WBA', 'TEL', 'RSG', 'GPN', 'IQV', 'TMUS', 'FTNT',
    'KHC', 'HLT', 'ES', 'FDX', 'DLTR', 'DDOG', 'EXR', 'AVB', 'EBAY', 'BIIB',
    'EFX', 'MTD', 'RMD', 'GWW', 'PPG', 'ROK', 'CPRT', 'GLW', 'FANG', 'HPQ'
]

ADDITIONAL_LARGE_CAPS = [
    'ROKU', 'UBER', 'LYFT', 'SNAP', 'SQ', 'SHOP', 'SPOT', 'ZOOM', 'DOCU',
    'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'COIN', 'RBLX', 'HOOD', 'DASH', 'ABNB',
    'PTON', 'ZM', 'OKTA', 'TWLO', 'SPLK', 'WDAY', 'VEEV', 'ZS', 'TEAM', 'ESTC',
    'MDB', 'FSLY', 'PINS', 'WORK', 'BILL', 'COUP', 'DOCU', 'GTLB', 'FVRR', 'UPWK'
]

SUSTAINABILITY_LEADERS = [
    'ENPH', 'SEDG', 'PLUG', 'FSLR', 'BE', 'NEP', 'BEP', 'ICLN', 'TAN',
    'QCLN', 'CTVA', 'DD', 'DOW', 'LYB', 'CF', 'FMC', 'MLM', 'VMC', 'NUE',
    'X', 'STLD', 'RS', 'CMC', 'WOR', 'PKG', 'IP', 'UFS', 'CCK', 'SON',
    'APD', 'LIN', 'ECL', 'EMN', 'PPG', 'SHW', 'RPM', 'AXTA', 'POL', 'KRA'
]

INTERNATIONAL_LEADERS = [
    'TSM', 'ASML', 'SAP', 'TM', 'SONY', 'NVO', 'MC', 'OR', 'RMS', 'CDI',
    'SHOP', 'SE', 'BABA', 'TSM', 'PDD', 'MELI', 'BIDU', 'JD', 'NTES', 'TME'
]

class EnhancedFinancialDataCollector:
    def __init__(self, config: Dict):        self.config = config
        self.api_keys = {
            'alpha_vantage': '82KM1CMCJ1SETUCA',
            'finnhub': 'd174edpr01qkv5jefmmgd174edpr01qkv5jefmn0',
            'quandl': config.get('api_keys', {}).get('quandl_api_key'),
            'polygon': config.get('api_keys', {}).get('polygon_api_key'),
            'iex_cloud': config.get('api_keys', {}).get('iex_cloud_api_key')
        }
        
        # Initialize API clients
        if self.api_keys['alpha_vantage']:
            self.av_fundamental = FundamentalData(key=self.api_keys['alpha_vantage'])
            self.av_timeseries = TimeSeries(key=self.api_keys['alpha_vantage'])
        
        if self.api_keys['finnhub']:
            self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
            
        self.logger = logging.getLogger(__name__)
        
    async def collect_comprehensive_financial_data(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Collect comprehensive financial data from multiple sources"""
        if symbols is None:
            symbols = list(set(S_P_500_COMPANIES + ADDITIONAL_LARGE_CAPS + SUSTAINABILITY_LEADERS + INTERNATIONAL_LEADERS))
        
        self.logger.info(f"Starting data collection for {len(symbols)} companies...")
        all_data = {}
        
        # Process in batches to avoid rate limits
        batch_size = self.config.get('dataset', {}).get('max_companies_per_batch', 50)
        rate_limit_delay = self.config.get('dataset', {}).get('rate_limit_delay', 1.0)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {len(batch)} companies")
            
            batch_data = await self._process_symbol_batch(batch)
            all_data.update(batch_data)
            
            # Rate limiting
            await asyncio.sleep(rate_limit_delay)
            
        self.logger.info(f"Data collection completed for {len(all_data)} companies")
        return all_data
    
    async def _process_symbol_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Process a batch of symbols concurrently"""
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def collect_with_semaphore(symbol):
            async with semaphore:
                return await self._collect_single_company_data(symbol)
        
        tasks = [collect_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_data = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception) and result:
                batch_data[symbol] = result
            else:
                self.logger.warning(f"Error collecting data for {symbol}: {result}")
                
        return batch_data
    
    async def _collect_single_company_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect comprehensive data for a single company"""
        try:
            company_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'financial_metrics': {},
                'fundamental_data': {},
                'market_data': {},
                'analyst_data': {},
                'news_sentiment': {},
                'technical_indicators': {},
                'industry_comparison': {},
                'data_quality_score': 0.0
            }
            
            # Yahoo Finance data (primary source - free and comprehensive)
            yf_data = await self._get_yahoo_finance_data(symbol)
            if yf_data:
                company_data['financial_metrics'].update(yf_data)
                company_data['data_quality_score'] += 0.4
            
            # Alpha Vantage data (fundamental analysis)
            if self.api_keys['alpha_vantage']:
                av_data = await self._get_alpha_vantage_data(symbol)
                if av_data:
                    company_data['fundamental_data'].update(av_data)
                    company_data['data_quality_score'] += 0.3
            
            # Finnhub data (news and sentiment)
            if self.api_keys['finnhub']:
                finnhub_data = await self._get_finnhub_data(symbol)
                if finnhub_data:
                    company_data['news_sentiment'].update(finnhub_data)
                    company_data['data_quality_score'] += 0.2
            
            # Calculate derived metrics
            calculated_metrics = self._calculate_comprehensive_metrics(company_data)
            company_data['calculated_metrics'] = calculated_metrics
            company_data['data_quality_score'] += 0.1
            
            # Only return if we have meaningful data
            if company_data['data_quality_score'] >= 0.3:
                return company_data
            else:
                self.logger.warning(f"Insufficient data quality for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    async def _get_yahoo_finance_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive Yahoo Finance data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Get key statistics and info
            info = ticker.info
            
            # Get historical data (5 years)
            hist_data = ticker.history(period="5y")
            
            # Get quarterly data
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow
            
            # Get recommendations and earnings
            recommendations = ticker.recommendations
            earnings = ticker.earnings
            
            return {
                'company_info': info,
                'income_statement': income_stmt.to_dict() if not income_stmt.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {},
                'quarterly_financials': quarterly_financials.to_dict() if not quarterly_financials.empty else {},
                'quarterly_balance_sheet': quarterly_balance_sheet.to_dict() if not quarterly_balance_sheet.empty else {},
                'quarterly_cashflow': quarterly_cashflow.to_dict() if not quarterly_cashflow.empty else {},
                'historical_prices': hist_data.to_dict() if not hist_data.empty else {},
                'recommendations': recommendations.to_dict() if recommendations is not None and not recommendations.empty else {},
                'earnings': earnings.to_dict() if earnings is not None and not earnings.empty else {},
                'key_metrics': self._extract_comprehensive_metrics(info, hist_data),
                'data_source': 'yahoo_finance',
                'collection_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return {}
    
    async def _get_alpha_vantage_data(self, symbol: str) -> Dict[str, Any]:
        """Get Alpha Vantage fundamental data"""
        try:
            av_data = {}
            
            # Company overview
            overview, _ = self.av_fundamental.get_company_overview(symbol)
            av_data['company_overview'] = overview
            
            # Income statement
            income_statement, _ = self.av_fundamental.get_income_statement_annual(symbol)
            av_data['income_statement_annual'] = income_statement
            
            # Balance sheet
            balance_sheet, _ = self.av_fundamental.get_balance_sheet_annual(symbol)
            av_data['balance_sheet_annual'] = balance_sheet
            
            # Cash flow
            cash_flow, _ = self.av_fundamental.get_cash_flow_annual(symbol)
            av_data['cash_flow_annual'] = cash_flow
            
            av_data['data_source'] = 'alpha_vantage'
            av_data['collection_timestamp'] = datetime.now().isoformat()
            
            return av_data
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return {}
    
    async def _get_finnhub_data(self, symbol: str) -> Dict[str, Any]:
        """Get Finnhub news and sentiment data"""
        try:
            finnhub_data = {}
            
            # Company profile
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            finnhub_data['company_profile'] = profile
            
            # News sentiment
            news_sentiment = self.finnhub_client.news_sentiment(symbol)
            finnhub_data['news_sentiment'] = news_sentiment
            
            # Company news
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            news = self.finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            finnhub_data['recent_news'] = news[:10]  # Limit to 10 most recent
            
            # Basic financials
            basic_financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            finnhub_data['basic_financials'] = basic_financials
            
            finnhub_data['data_source'] = 'finnhub'
            finnhub_data['collection_timestamp'] = datetime.now().isoformat()
            
            return finnhub_data
            
        except Exception as e:
            self.logger.error(f"Finnhub error for {symbol}: {e}")
            return {}
    
    def _extract_comprehensive_metrics(self, info: Dict, hist_data: pd.DataFrame) -> Dict[str, float]:
        """Extract comprehensive financial metrics"""
        metrics = {}
        
        # Basic company info
        metrics['market_cap'] = info.get('marketCap', 0)
        metrics['enterprise_value'] = info.get('enterpriseValue', 0)
        metrics['shares_outstanding'] = info.get('sharesOutstanding', 0)
        metrics['float_shares'] = info.get('floatShares', 0)
        
        # Profitability metrics
        metrics['profit_margin'] = info.get('profitMargins', 0)
        metrics['operating_margin'] = info.get('operatingMargins', 0)
        metrics['gross_margin'] = info.get('grossMargins', 0)
        metrics['ebitda_margin'] = info.get('ebitdaMargins', 0)
        metrics['return_on_equity'] = info.get('returnOnEquity', 0)
        metrics['return_on_assets'] = info.get('returnOnAssets', 0)
        
        # Valuation metrics
        metrics['pe_ratio'] = info.get('trailingPE', 0)
        metrics['forward_pe'] = info.get('forwardPE', 0)
        metrics['peg_ratio'] = info.get('pegRatio', 0)
        metrics['price_to_book'] = info.get('priceToBook', 0)
        metrics['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
        metrics['ev_to_revenue'] = info.get('enterpriseToRevenue', 0)
        metrics['ev_to_ebitda'] = info.get('enterpriseToEbitda', 0)
        
        # Financial health metrics
        metrics['debt_to_equity'] = info.get('debtToEquity', 0)
        metrics['current_ratio'] = info.get('currentRatio', 0)
        metrics['quick_ratio'] = info.get('quickRatio', 0)
        metrics['cash_per_share'] = info.get('totalCashPerShare', 0)
        metrics['book_value_per_share'] = info.get('bookValue', 0)
        
        # Growth metrics
        metrics['revenue_growth'] = info.get('revenueGrowth', 0)
        metrics['earnings_growth'] = info.get('earningsGrowth', 0)
        metrics['revenue_per_share'] = info.get('revenuePerShare', 0)
        
        # Dividend metrics
        metrics['dividend_yield'] = info.get('dividendYield', 0)
        metrics['payout_ratio'] = info.get('payoutRatio', 0)
        metrics['dividend_rate'] = info.get('dividendRate', 0)
        
        # Market metrics
        metrics['beta'] = info.get('beta', 0)
        metrics['52_week_high'] = info.get('fiftyTwoWeekHigh', 0)
        metrics['52_week_low'] = info.get('fiftyTwoWeekLow', 0)
        
        # Calculate additional metrics from historical data
        if not hist_data.empty and len(hist_data) > 0:
            metrics.update(self._calculate_technical_metrics(hist_data))
            
        return metrics
    
    def _calculate_technical_metrics(self, hist_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis metrics"""
        metrics = {}
        
        if len(hist_data) < 20:
            return metrics
            
        try:
            # Price volatility
            returns = hist_data['Close'].pct_change().dropna()
            metrics['volatility_30d'] = returns.tail(30).std() * np.sqrt(252)
            metrics['volatility_1y'] = returns.std() * np.sqrt(252)
            
            # Moving averages
            metrics['sma_20'] = hist_data['Close'].tail(20).mean()
            metrics['sma_50'] = hist_data['Close'].tail(50).mean() if len(hist_data) >= 50 else 0
            metrics['sma_200'] = hist_data['Close'].tail(200).mean() if len(hist_data) >= 200 else 0
            
            # Price performance
            current_price = hist_data['Close'].iloc[-1]
            metrics['price_change_1m'] = (current_price / hist_data['Close'].iloc[-20] - 1) if len(hist_data) >= 20 else 0
            metrics['price_change_3m'] = (current_price / hist_data['Close'].iloc[-60] - 1) if len(hist_data) >= 60 else 0
            metrics['price_change_1y'] = (current_price / hist_data['Close'].iloc[-252] - 1) if len(hist_data) >= 252 else 0
            
            # Volume metrics
            metrics['avg_volume_30d'] = hist_data['Volume'].tail(30).mean()
            metrics['volume_trend'] = hist_data['Volume'].tail(10).mean() / hist_data['Volume'].tail(30).mean()
            
        except Exception as e:
            self.logger.error(f"Technical metrics calculation error: {e}")
            
        return metrics
    
    def _calculate_comprehensive_metrics(self, company_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive derived metrics"""
        calculated = {}
        
        try:
            financial_metrics = company_data.get('financial_metrics', {})
            key_metrics = financial_metrics.get('key_metrics', {})
            
            # Financial strength score
            calculated['financial_strength_score'] = self._calculate_financial_strength(key_metrics)
            
            # Growth score
            calculated['growth_score'] = self._calculate_growth_score(key_metrics)
            
            # Value score
            calculated['value_score'] = self._calculate_value_score(key_metrics)
            
            # Quality score
            calculated['quality_score'] = self._calculate_quality_score(key_metrics)
            
            # Composite investment score
            calculated['investment_score'] = np.mean([
                calculated['financial_strength_score'],
                calculated['growth_score'],
                calculated['value_score'],
                calculated['quality_score']
            ])
            
        except Exception as e:
            self.logger.error(f"Comprehensive metrics calculation error: {e}")
            
        return calculated
    
    def _calculate_financial_strength(self, metrics: Dict) -> float:
        """Calculate financial strength score (0-100)"""
        score = 0
        factors = 0
        
        # Current ratio (good if > 1.5)
        if metrics.get('current_ratio', 0) > 1.5:
            score += 25
        elif metrics.get('current_ratio', 0) > 1.0:
            score += 15
        factors += 1
        
        # Debt to equity (good if < 0.5)
        debt_equity = metrics.get('debt_to_equity', 0)
        if debt_equity < 0.3:
            score += 25
        elif debt_equity < 0.7:
            score += 15
        factors += 1
        
        # ROE (good if > 15%)
        roe = metrics.get('return_on_equity', 0)
        if roe > 0.15:
            score += 25
        elif roe > 0.10:
            score += 15
        factors += 1
        
        # Operating margin (good if > 10%)
        op_margin = metrics.get('operating_margin', 0)
        if op_margin > 0.15:
            score += 25
        elif op_margin > 0.10:
            score += 15
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate growth score (0-100)"""
        score = 0
        factors = 0
        
        # Revenue growth
        revenue_growth = metrics.get('revenue_growth', 0)
        if revenue_growth > 0.20:
            score += 30
        elif revenue_growth > 0.10:
            score += 20
        elif revenue_growth > 0.05:
            score += 10
        factors += 1
        
        # Earnings growth
        earnings_growth = metrics.get('earnings_growth', 0)
        if earnings_growth > 0.20:
            score += 30
        elif earnings_growth > 0.10:
            score += 20
        elif earnings_growth > 0.05:
            score += 10
        factors += 1
        
        # PEG ratio (good if < 1.5)
        peg = metrics.get('peg_ratio', 0)
        if 0 < peg < 1.0:
            score += 25
        elif 1.0 <= peg < 1.5:
            score += 15
        factors += 1
        
        # Price performance
        price_1y = metrics.get('price_change_1y', 0)
        if price_1y > 0.20:
            score += 15
        elif price_1y > 0.10:
            score += 10
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _calculate_value_score(self, metrics: Dict) -> float:
        """Calculate value score (0-100)"""
        score = 0
        factors = 0
        
        # P/E ratio (good if reasonable)
        pe = metrics.get('pe_ratio', 0)
        if 10 <= pe <= 20:
            score += 30
        elif 5 <= pe < 10 or 20 < pe <= 25:
            score += 20
        elif pe > 0:
            score += 10
        factors += 1
        
        # Price to book (good if < 3)
        pb = metrics.get('price_to_book', 0)
        if 0 < pb < 1.5:
            score += 25
        elif 1.5 <= pb < 3.0:
            score += 15
        factors += 1
        
        # Price to sales (good if < 3)
        ps = metrics.get('price_to_sales', 0)
        if 0 < ps < 2.0:
            score += 25
        elif 2.0 <= ps < 4.0:
            score += 15
        factors += 1
        
        # EV/EBITDA (good if < 15)
        ev_ebitda = metrics.get('ev_to_ebitda', 0)
        if 0 < ev_ebitda < 10:
            score += 20
        elif 10 <= ev_ebitda < 15:
            score += 10
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate quality score (0-100)"""
        score = 0
        factors = 0
        
        # Profit margin (good if > 10%)
        profit_margin = metrics.get('profit_margin', 0)
        if profit_margin > 0.15:
            score += 30
        elif profit_margin > 0.10:
            score += 20
        elif profit_margin > 0.05:
            score += 10
        factors += 1
        
        # ROA (good if > 8%)
        roa = metrics.get('return_on_assets', 0)
        if roa > 0.12:
            score += 25
        elif roa > 0.08:
            score += 15
        factors += 1
        
        # Gross margin (good if > 40%)
        gross_margin = metrics.get('gross_margin', 0)
        if gross_margin > 0.50:
            score += 25
        elif gross_margin > 0.40:
            score += 15
        factors += 1
        
        # Beta (good if between 0.8-1.2)
        beta = metrics.get('beta', 0)
        if 0.8 <= beta <= 1.2:
            score += 20
        elif 0.5 <= beta < 0.8 or 1.2 < beta <= 1.5:
            score += 10
        factors += 1
        
        return score / factors if factors > 0 else 0

def get_all_target_companies() -> List[str]:
    """Get comprehensive list of all target companies for maximum dataset"""
    return list(set(S_P_500_COMPANIES + ADDITIONAL_LARGE_CAPS + SUSTAINABILITY_LEADERS + INTERNATIONAL_LEADERS))
