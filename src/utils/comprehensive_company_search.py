"""
Comprehensive Company Search Engine

This module searches for companies based on user criteria using multiple data sources:
- Research papers and academic databases
- Financial APIs (Alpha Vantage, Finnhub, etc.)
- Company registries and databases
- Google Gemini 2.5 Pro for intelligent search
"""

import os
import asyncio
import aiohttp
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google Gemini integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Financial APIs
try:
    import yfinance as yf
    import alpha_vantage
    import finnhub
    FINANCIAL_APIS_AVAILABLE = True
except ImportError:
    FINANCIAL_APIS_AVAILABLE = False

# Research databases
try:
    import arxiv
    import scholarly
    RESEARCH_APIS_AVAILABLE = True
except ImportError:
    RESEARCH_APIS_AVAILABLE = False

class ComprehensiveCompanySearchEngine:
    """
    Advanced company search engine that finds companies based on detailed criteria
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the search engine"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize APIs
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # Initialize Gemini
        if GEMINI_AVAILABLE and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.gemini_available = True
            self.logger.info("Google Gemini 2.5 Pro initialized")
        else:
            self.gemini_available = False
            self.logger.warning("Gemini API not available")
        
        # Company databases and sources
        self.data_sources = {
            'financial_apis': FINANCIAL_APIS_AVAILABLE,
            'research_papers': RESEARCH_APIS_AVAILABLE,
            'company_registries': True,  # We have local registry data
            'gemini_search': self.gemini_available
        }
        
        # Load existing company data
        self.company_database = self._load_company_database()
        
        self.logger.info("Company Search Engine initialized")
    
    def _load_company_database(self) -> pd.DataFrame:
        """Load existing company database from project data"""
        try:
            # Try to load from comprehensive dataset
            data_path = Path("data/comprehensive")
            possible_files = [
                "ultimate_dataset.csv",
                "enhanced_multi_api_dataset.csv",
                "spr_analysis_dataset.csv"
            ]
            
            for file_name in possible_files:
                file_path = data_path / file_name
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    self.logger.info(f"Loaded company database: {file_name} ({len(df)} companies)")
                    return df
            
            # If no files found, create empty dataframe with expected columns
            self.logger.warning("No existing company database found, creating empty one")
            return pd.DataFrame(columns=[
                'symbol', 'name', 'sector', 'industry', 'location', 'market_cap',
                'revenue', 'profit_margin', 'spr_score', 'sustainability_score',
                'esg_score', 'risk_level'
            ])
            
        except Exception as e:
            self.logger.error(f"Error loading company database: {e}")
            return pd.DataFrame()
    
    async def search_companies_by_criteria(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main search function that finds companies based on user criteria
        
        Args:
            search_criteria: Dictionary containing search parameters
            
        Returns:
            List of companies matching the criteria
        """
        self.logger.info(f"Starting comprehensive company search with criteria: {search_criteria}")
        
        # Step 1: Search existing database
        database_results = await self._search_database(search_criteria)
        
        # Step 2: Search using financial APIs
        api_results = await self._search_financial_apis(search_criteria)
        
        # Step 3: Search research papers for emerging companies
        research_results = await self._search_research_papers(search_criteria)
        
        # Step 4: Use Gemini AI for intelligent search
        gemini_results = await self._search_with_gemini(search_criteria)
        
        # Step 5: Combine and deduplicate results
        all_results = self._combine_search_results(
            database_results, api_results, research_results, gemini_results
        )
        
        # Step 6: Enrich results with additional data
        enriched_results = await self._enrich_company_data(all_results)
        
        self.logger.info(f"Found {len(enriched_results)} companies matching criteria")
        return enriched_results
    
    async def _search_database(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search existing company database"""
        try:
            df = self.company_database.copy()
            
            if df.empty:
                return []
            
            # Apply filters based on criteria
            if criteria.get('location'):
                locations = [loc.lower() for loc in criteria['location']]
                df = df[df['location'].str.lower().isin(locations)]
            
            if criteria.get('sectors'):
                df = df[df['sector'].isin(criteria['sectors'])]
            
            if criteria.get('industries'):
                df = df[df['industry'].isin(criteria['industries'])]
            
            if criteria.get('market_cap_min'):
                df = df[df['market_cap'] >= criteria['market_cap_min']]
            
            if criteria.get('market_cap_max'):
                df = df[df['market_cap'] <= criteria['market_cap_max']]
            
            if criteria.get('profit_level'):
                profit_level = criteria['profit_level'].lower()
                if profit_level == 'high':
                    df = df[df['profit_margin'] >= 20]
                elif profit_level == 'medium':
                    df = df[(df['profit_margin'] >= 10) & (df['profit_margin'] < 20)]
                elif profit_level == 'low':
                    df = df[df['profit_margin'] < 10]
            
            # Convert to list of dictionaries
            results = df.to_dict('records')
            
            # Add source information
            for result in results:
                result['data_source'] = 'internal_database'
                result['confidence_score'] = 0.9  # High confidence for existing data
            
            self.logger.info(f"Database search found {len(results)} companies")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in database search: {e}")
            return []
    
    async def _search_financial_apis(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search using financial APIs"""
        if not FINANCIAL_APIS_AVAILABLE:
            return []
        
        try:
            results = []
            
            # Search using sector/industry information
            if criteria.get('sectors') or criteria.get('industries'):
                # Use yfinance to get sector/industry company lists
                sector_results = await self._search_by_sector(criteria)
                results.extend(sector_results)
            
            # Search by market cap range
            if criteria.get('market_cap_min') or criteria.get('market_cap_max'):
                market_cap_results = await self._search_by_market_cap(criteria)
                results.extend(market_cap_results)
            
            # Add source information
            for result in results:
                result['data_source'] = 'financial_apis'
                result['confidence_score'] = 0.8
            
            self.logger.info(f"Financial APIs search found {len(results)} companies")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in financial APIs search: {e}")
            return []
    
    async def _search_by_sector(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search companies by sector using financial APIs"""
        results = []
        
        # Define sector to ticker mappings (major companies)
        sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'DHR', 'BMY', 'ABBV', 'MDT'],
            'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'BRK-B'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE'],
            'Consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST'],
            'Manufacturing': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UTX', 'LMT', 'RTX', 'DE', 'EMR']
        }
        
        sectors = criteria.get('sectors', [])
        
        for sector in sectors:
            if sector in sector_tickers:
                for ticker in sector_tickers[sector]:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        if info:
                            company_data = {
                                'symbol': ticker,
                                'name': info.get('longName', ticker),
                                'sector': info.get('sector', sector),
                                'industry': info.get('industry', 'Unknown'),
                                'location': info.get('country', 'Unknown'),
                                'market_cap': info.get('marketCap', 0),
                                'revenue': info.get('totalRevenue', 0),
                                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                                'pe_ratio': info.get('trailingPE', 0),
                                'beta': info.get('beta', 1.0),
                                'esg_score': info.get('esgScores', {}).get('totalEsg', 'N/A') if info.get('esgScores') else 'N/A'
                            }
                            results.append(company_data)
                            
                    except Exception as e:
                        self.logger.debug(f"Error fetching data for {ticker}: {e}")
                        continue
        
        return results
    
    async def _search_by_market_cap(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search companies by market cap range"""
        results = []
        
        # Define market cap categories with representative tickers
        market_cap_tickers = {
            'large': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ'],
            'mid': ['AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'TMO', 'COST'],
            'small': ['ROKU', 'PINS', 'SNAP', 'TWTR', 'SPOT', 'SQ', 'ZM', 'PTON', 'DOCU', 'CRWD']
        }
        
        min_cap = criteria.get('market_cap_min', 0)
        max_cap = criteria.get('market_cap_max', float('inf'))
        
        # Determine which categories to search
        categories_to_search = []
        if min_cap < 2000000000:  # < 2B
            categories_to_search.append('small')
        if min_cap < 10000000000 and max_cap > 2000000000:  # 2B - 10B
            categories_to_search.append('mid')
        if max_cap > 10000000000:  # > 10B
            categories_to_search.append('large')
        
        for category in categories_to_search:
            for ticker in market_cap_tickers[category]:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if info and info.get('marketCap'):
                        market_cap = info['marketCap']
                        
                        # Check if within range
                        if min_cap <= market_cap <= max_cap:
                            company_data = {
                                'symbol': ticker,
                                'name': info.get('longName', ticker),
                                'sector': info.get('sector', 'Unknown'),
                                'industry': info.get('industry', 'Unknown'),
                                'location': info.get('country', 'Unknown'),
                                'market_cap': market_cap,
                                'revenue': info.get('totalRevenue', 0),
                                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
                            }
                            results.append(company_data)
                            
                except Exception as e:
                    self.logger.debug(f"Error fetching market cap data for {ticker}: {e}")
                    continue
        
        return results
    
    async def _search_research_papers(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search research papers for emerging companies and trends"""
        if not RESEARCH_APIS_AVAILABLE:
            return []
        
        try:
            results = []
            
            # Create search queries based on criteria
            search_queries = self._generate_research_queries(criteria)
            
            for query in search_queries:
                # Search ArXiv for relevant papers
                papers = await self._search_arxiv(query)
                
                # Extract company mentions from papers
                companies = self._extract_companies_from_papers(papers)
                results.extend(companies)
            
            # Add source information
            for result in results:
                result['data_source'] = 'research_papers'
                result['confidence_score'] = 0.6  # Lower confidence for research-based findings
            
            self.logger.info(f"Research papers search found {len(results)} companies")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in research papers search: {e}")
            return []
    
    def _generate_research_queries(self, criteria: Dict[str, Any]) -> List[str]:
        """Generate research search queries based on criteria"""
        queries = []
        
        # Base terms
        base_terms = ['companies', 'corporations', 'enterprises', 'firms']
        
        # Add sector-specific terms
        if criteria.get('sectors'):
            for sector in criteria['sectors']:
                for term in base_terms:
                    queries.append(f"{sector.lower()} {term}")
        
        # Add location-specific terms
        if criteria.get('location'):
            for location in criteria['location']:
                for term in base_terms:
                    queries.append(f"{location.lower()} {term}")
        
        # Add profit/performance terms
        if criteria.get('profit_level'):
            profit_terms = {
                'high': ['profitable', 'high-performing', 'successful'],
                'medium': ['growing', 'emerging', 'developing'],
                'low': ['startup', 'early-stage', 'new']
            }
            level = criteria['profit_level'].lower()
            if level in profit_terms:
                for profit_term in profit_terms[level]:
                    for term in base_terms:
                        queries.append(f"{profit_term} {term}")
        
        # Add sustainability terms if relevant
        sustainability_queries = [
            'sustainable companies',
            'ESG companies',
            'green companies',
            'renewable energy companies',
            'clean technology companies'
        ]
        queries.extend(sustainability_queries)
        
        return queries[:10]  # Limit to 10 queries to avoid rate limits
    
    async def _search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """Search ArXiv for research papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [author.name for author in result.authors],
                    'published': result.published,
                    'url': result.entry_id
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _extract_companies_from_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract company mentions from research papers"""
        companies = []
        
        # Common company patterns
        company_patterns = [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|Ltd|Limited|Company|Co)\b',
            r'\b[A-Z]{2,5}\b',  # Stock tickers
            r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b'  # Company names
        ]
        
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            
            for pattern in company_patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    # Filter out common false positives
                    if self._is_likely_company(match):
                        company_data = {
                            'name': match,
                            'symbol': 'Unknown',
                            'sector': 'Unknown',
                            'industry': 'Unknown',
                            'location': 'Unknown',
                            'market_cap': 0,
                            'source_paper': paper['title'],
                            'source_url': paper['url']
                        }
                        companies.append(company_data)
        
        return companies
    
    def _is_likely_company(self, text: str) -> bool:
        """Check if text is likely a company name"""
        # Filter out common false positives
        false_positives = {
            'USA', 'EU', 'UK', 'US', 'CEO', 'CTO', 'CFO', 'AI', 'ML', 'IT', 'API',
            'GDP', 'ROI', 'ESG', 'CSR', 'IPO', 'NYSE', 'NASDAQ', 'SEC', 'FDA'
        }
        
        if text.upper() in false_positives:
            return False
        
        # Must be reasonable length
        if len(text) < 2 or len(text) > 50:
            return False
        
        return True
    
    async def _search_with_gemini(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use Google Gemini 2.5 Pro for intelligent company search"""
        if not self.gemini_available:
            return []
        
        try:
            # Create detailed prompt for Gemini
            prompt = self._create_gemini_search_prompt(criteria)
            
            response = await self.gemini_model.generate_content_async(prompt)
            
            if response and response.text:
                # Parse Gemini response to extract company information
                companies = self._parse_gemini_response(response.text)
                
                # Add source information
                for company in companies:
                    company['data_source'] = 'gemini_ai'
                    company['confidence_score'] = 0.8
                
                self.logger.info(f"Gemini AI search found {len(companies)} companies")
                return companies
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in Gemini search: {e}")
            return []
    
    def _create_gemini_search_prompt(self, criteria: Dict[str, Any]) -> str:
        """Create a detailed prompt for Gemini to search for companies"""
        prompt = f"""
        As a financial research expert, find companies that match the following criteria:

        SEARCH CRITERIA:
        """
        
        if criteria.get('location'):
            prompt += f"- Geographic Location: {', '.join(criteria['location'])}\n"
        
        if criteria.get('sectors'):
            prompt += f"- Industry Sectors: {', '.join(criteria['sectors'])}\n"
        
        if criteria.get('industries'):
            prompt += f"- Specific Industries: {', '.join(criteria['industries'])}\n"
        
        if criteria.get('market_cap_min') or criteria.get('market_cap_max'):
            min_cap = criteria.get('market_cap_min', 0)
            max_cap = criteria.get('market_cap_max', 'unlimited')
            prompt += f"- Market Capitalization: ${min_cap:,} to {max_cap if isinstance(max_cap, str) else f'${max_cap:,}'}\n"
        
        if criteria.get('profit_level'):
            prompt += f"- Profit Level: {criteria['profit_level']} profit companies\n"
        
        if criteria.get('company_size'):
            prompt += f"- Company Size: {criteria['company_size']}\n"
        
        if criteria.get('additional_criteria'):
            prompt += f"- Additional Requirements: {criteria['additional_criteria']}\n"
        
        prompt += f"""
        
        Please provide a list of 15-20 companies that match these criteria. For each company, provide:

        1. Company Name
        2. Stock Symbol (if publicly traded)
        3. Industry/Sector
        4. Country/Location
        5. Approximate Market Cap
        6. Brief description of why it matches the criteria
        7. Profit level (High/Medium/Low)
        8. ESG/Sustainability rating (if known)

        Format the response as JSON with the following structure:
        {{
            "companies": [
                {{
                    "name": "Company Name",
                    "symbol": "SYMBOL",
                    "sector": "Sector",
                    "industry": "Industry",
                    "location": "Country",
                    "market_cap": 1000000000,
                    "description": "Why it matches criteria",
                    "profit_level": "High/Medium/Low",
                    "esg_score": "A/B/C or rating",
                    "match_score": 0.95
                }}
            ]
        }}

        Focus on companies that are well-established, publicly traded when possible, and have good financial data available.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Gemini's response to extract company information"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                
                if 'companies' in data:
                    companies = data['companies']
                    
                    # Validate and clean the data
                    validated_companies = []
                    for company in companies:
                        if self._validate_company_data(company):
                            validated_companies.append(company)
                    
                    return validated_companies
            
            # If JSON parsing fails, try to extract information manually
            return self._extract_companies_from_text(response_text)
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            return []
    
    def _validate_company_data(self, company: Dict[str, Any]) -> bool:
        """Validate company data from Gemini response"""
        required_fields = ['name']
        
        for field in required_fields:
            if not company.get(field):
                return False
        
        # Set defaults for missing fields
        company.setdefault('symbol', 'Unknown')
        company.setdefault('sector', 'Unknown')
        company.setdefault('industry', 'Unknown')
        company.setdefault('location', 'Unknown')
        company.setdefault('market_cap', 0)
        company.setdefault('profit_level', 'Unknown')
        company.setdefault('esg_score', 'Unknown')
        company.setdefault('match_score', 0.5)
        
        return True
    
    def _extract_companies_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract company information from plain text response"""
        companies = []
        
        # Look for company names in the text
        lines = text.split('\n')
        current_company = {}
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_company and current_company.get('name'):
                    companies.append(current_company.copy())
                    current_company = {}
                continue
            
            # Try to identify company information patterns
            if any(word in line.lower() for word in ['company', 'corp', 'inc', 'ltd']):
                current_company['name'] = line
            elif 'symbol:' in line.lower():
                current_company['symbol'] = line.split(':')[-1].strip()
            elif 'sector:' in line.lower():
                current_company['sector'] = line.split(':')[-1].strip()
            elif 'location:' in line.lower():
                current_company['location'] = line.split(':')[-1].strip()
        
        # Add final company if exists
        if current_company and current_company.get('name'):
            companies.append(current_company)
        
        return companies
    
    def _combine_search_results(self, *result_lists) -> List[Dict[str, Any]]:
        """Combine and deduplicate search results from multiple sources"""
        all_results = []
        seen_companies = set()
        
        for result_list in result_lists:
            for company in result_list:
                # Create a unique identifier
                identifier = self._create_company_identifier(company)
                
                if identifier not in seen_companies:
                    seen_companies.add(identifier)
                    all_results.append(company)
                else:
                    # If we've seen this company, merge additional data
                    existing_company = next(
                        (c for c in all_results if self._create_company_identifier(c) == identifier), 
                        None
                    )
                    if existing_company:
                        self._merge_company_data(existing_company, company)
        
        return all_results
    
    def _create_company_identifier(self, company: Dict[str, Any]) -> str:
        """Create a unique identifier for a company"""
        name = company.get('name', '').lower().strip()
        symbol = company.get('symbol', '').upper().strip()
        
        # Clean up name
        name = re.sub(r'\b(inc|corp|corporation|ltd|limited|company|co)\b', '', name)
        name = re.sub(r'[^\w\s]', '', name).strip()
        
        # Use symbol if available, otherwise use cleaned name
        if symbol and symbol != 'UNKNOWN':
            return symbol
        else:
            return name
    
    def _merge_company_data(self, existing: Dict[str, Any], new: Dict[str, Any]):
        """Merge data from multiple sources for the same company"""
        for key, value in new.items():
            if key not in existing or existing[key] in ['Unknown', '', 0, None]:
                existing[key] = value
            elif key == 'confidence_score':
                # Average confidence scores
                existing[key] = (existing.get(key, 0) + value) / 2
    
    async def _enrich_company_data(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich company data with additional information"""
        enriched_companies = []
        
        for company in companies:
            try:
                # Try to get additional data from yfinance if symbol is available
                symbol = company.get('symbol')
                if symbol and symbol != 'Unknown':
                    enriched_data = await self._enrich_with_yfinance(company, symbol)
                    if enriched_data:
                        company.update(enriched_data)
                
                # Calculate preliminary SPR score
                company['preliminary_spr_score'] = self._calculate_preliminary_spr(company)
                
                # Add timestamp
                company['search_timestamp'] = datetime.now().isoformat()
                
                enriched_companies.append(company)
                
            except Exception as e:
                self.logger.debug(f"Error enriching company {company.get('name', 'Unknown')}: {e}")
                enriched_companies.append(company)  # Add anyway with available data
        
        return enriched_companies
    
    async def _enrich_with_yfinance(self, company: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enrich company data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return {}
            
            enriched_data = {}
            
            # Update basic information
            if info.get('longName'):
                enriched_data['name'] = info['longName']
            if info.get('sector'):
                enriched_data['sector'] = info['sector']
            if info.get('industry'):
                enriched_data['industry'] = info['industry']
            if info.get('country'):
                enriched_data['location'] = info['country']
            
            # Financial metrics
            enriched_data['market_cap'] = info.get('marketCap', 0)
            enriched_data['revenue'] = info.get('totalRevenue', 0)
            enriched_data['profit_margin'] = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
            enriched_data['pe_ratio'] = info.get('trailingPE', 0)
            enriched_data['beta'] = info.get('beta', 1.0)
            enriched_data['debt_to_equity'] = info.get('debtToEquity', 0)
            enriched_data['roe'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            
            # ESG data if available
            if info.get('esgScores'):
                enriched_data['esg_score'] = info['esgScores'].get('totalEsg', 'N/A')
                enriched_data['environmental_score'] = info['esgScores'].get('environmentScore', 0)
                enriched_data['social_score'] = info['esgScores'].get('socialScore', 0)
                enriched_data['governance_score'] = info['esgScores'].get('governanceScore', 0)
            
            return enriched_data
            
        except Exception as e:
            self.logger.debug(f"Error enriching with yfinance for {symbol}: {e}")
            return {}
    
    def _calculate_preliminary_spr(self, company: Dict[str, Any]) -> float:
        """Calculate a preliminary SPR score for ranking"""
        try:
            # Base score components
            profit_score = 0
            sustainability_score = 0
            financial_health_score = 0
            
            # Profit performance (0-10 scale)
            profit_margin = company.get('profit_margin', 0)
            if profit_margin >= 30:
                profit_score = 10
            elif profit_margin >= 20:
                profit_score = 8
            elif profit_margin >= 10:
                profit_score = 6
            elif profit_margin >= 5:
                profit_score = 4
            elif profit_margin > 0:
                profit_score = 2
            
            # Sustainability score (0-10 scale)
            esg_score = company.get('esg_score')
            if isinstance(esg_score, (int, float)):
                sustainability_score = min(esg_score / 10, 10)  # Normalize to 0-10
            elif esg_score == 'A+':
                sustainability_score = 10
            elif esg_score == 'A':
                sustainability_score = 8.5
            elif esg_score == 'A-':
                sustainability_score = 7.5
            elif esg_score == 'B+':
                sustainability_score = 6.5
            elif esg_score == 'B':
                sustainability_score = 5.5
            elif esg_score == 'B-':
                sustainability_score = 4.5
            else:
                sustainability_score = 5  # Default neutral score
            
            # Financial health score (0-10 scale)
            market_cap = company.get('market_cap', 0)
            pe_ratio = company.get('pe_ratio', 0)
            debt_to_equity = company.get('debt_to_equity', 0)
            
            if market_cap > 100000000000:  # >100B
                financial_health_score += 3
            elif market_cap > 10000000000:  # >10B
                financial_health_score += 2.5
            elif market_cap > 1000000000:  # >1B
                financial_health_score += 2
            else:
                financial_health_score += 1
            
            if 10 <= pe_ratio <= 25:  # Reasonable P/E ratio
                financial_health_score += 2
            elif pe_ratio > 0:
                financial_health_score += 1
            
            if debt_to_equity < 0.5:  # Low debt
                financial_health_score += 2
            elif debt_to_equity < 1.0:
                financial_health_score += 1
            
            # Confidence adjustment
            confidence = company.get('confidence_score', 0.5)
            
            # Calculate weighted SPR score
            spr_score = (
                profit_score * 0.4 +
                sustainability_score * 0.4 +
                financial_health_score * 0.2
            ) * confidence
            
            return round(min(spr_score, 10), 2)
            
        except Exception as e:
            self.logger.debug(f"Error calculating preliminary SPR: {e}")
            return 5.0  # Default score
    
    def get_search_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of search results"""
        if not results:
            return {"total_companies": 0, "message": "No companies found"}
        
        summary = {
            "total_companies": len(results),
            "data_sources": {},
            "sectors": {},
            "locations": {},
            "avg_spr_score": 0,
            "top_performers": []
        }
        
        # Count by data source
        for company in results:
            source = company.get('data_source', 'unknown')
            summary["data_sources"][source] = summary["data_sources"].get(source, 0) + 1
        
        # Count by sector
        for company in results:
            sector = company.get('sector', 'Unknown')
            summary["sectors"][sector] = summary["sectors"].get(sector, 0) + 1
        
        # Count by location
        for company in results:
            location = company.get('location', 'Unknown')
            summary["locations"][location] = summary["locations"].get(location, 0) + 1
        
        # Calculate average SPR score
        spr_scores = [c.get('preliminary_spr_score', 0) for c in results if c.get('preliminary_spr_score')]
        if spr_scores:
            summary["avg_spr_score"] = round(sum(spr_scores) / len(spr_scores), 2)
        
        # Get top performers
        sorted_companies = sorted(results, key=lambda x: x.get('preliminary_spr_score', 0), reverse=True)
        summary["top_performers"] = [
            {
                "name": c.get('name', 'Unknown'),
                "symbol": c.get('symbol', 'N/A'),
                "spr_score": c.get('preliminary_spr_score', 0)
            }
            for c in sorted_companies[:5]
        ]
        
        return summary
