import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
import json
import re
from datetime import datetime, timedelta
import logging
from urllib.parse import urljoin, urlparse
import time

class ESGDataCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.esg_sources = {
            'msci': 'https://www.msci.com/esg-ratings',
            'sustainalytics': 'https://www.sustainalytics.com',
            'refinitiv': 'https://www.refinitiv.com/en/sustainable-finance',
            'bloomberg': 'https://www.bloomberg.com/professional/product/esg-data/',
            'cdp': 'https://www.cdp.net'
        }
        
        # ESG keywords for content analysis
        self.environmental_keywords = [
            'carbon emission', 'greenhouse gas', 'renewable energy', 'solar', 'wind',
            'water usage', 'waste reduction', 'recycling', 'circular economy',
            'biodiversity', 'deforestation', 'climate change', 'sustainability',
            'energy efficiency', 'clean technology', 'environmental impact'
        ]
        
        self.social_keywords = [
            'diversity', 'inclusion', 'employee satisfaction', 'workplace safety',
            'human rights', 'community investment', 'philanthropy', 'education',
            'healthcare access', 'fair labor', 'supply chain ethics', 'customer satisfaction',
            'data privacy', 'product safety', 'social impact', 'stakeholder engagement'
        ]
        
        self.governance_keywords = [
            'board independence', 'board diversity', 'executive compensation',
            'shareholder rights', 'transparency', 'business ethics', 'anti-corruption',
            'risk management', 'cybersecurity', 'regulatory compliance',
            'audit committee', 'whistleblower', 'code of conduct', 'governance structure'
        ]
    
    async def collect_comprehensive_esg_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Collect comprehensive ESG data from multiple sources"""
        self.logger.info(f"Starting ESG data collection for {len(symbols)} companies...")
        esg_data = {}
        
        # Process in batches to avoid overwhelming servers
        batch_size = 20
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            self.logger.info(f"Processing ESG batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            batch_tasks = [self._collect_company_esg(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if not isinstance(result, Exception) and result:
                    esg_data[symbol] = result
                else:
                    self.logger.warning(f"ESG data collection failed for {symbol}: {result}")
            
            # Rate limiting
            await asyncio.sleep(2)
        
        self.logger.info(f"ESG data collection completed for {len(esg_data)} companies")
        return esg_data
    
    async def _collect_company_esg(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect comprehensive ESG data for a single company"""
        try:
            esg_data = {
                'symbol': symbol,
                'collection_timestamp': datetime.now().isoformat(),
                'environmental_score': 0,
                'social_score': 0,
                'governance_score': 0,
                'overall_esg_score': 0,
                'carbon_footprint': {},
                'sustainability_initiatives': [],
                'governance_metrics': {},
                'social_impact_metrics': {},
                'esg_ratings': {},
                'sustainability_reports': [],
                'esg_news_sentiment': {},
                'data_quality_score': 0.0
            }
            
            # Get Yahoo Finance ESG data
            yf_esg = await self._get_yahoo_esg_data(symbol)
            if yf_esg:
                esg_data.update(yf_esg)
                esg_data['data_quality_score'] += 0.3
            
            # Scrape company sustainability reports
            sustainability_reports = await self._scrape_sustainability_reports(symbol)
            if sustainability_reports:
                esg_data['sustainability_reports'] = sustainability_reports
                esg_data['data_quality_score'] += 0.2
            
            # Analyze ESG content from various sources
            content_analysis = await self._analyze_esg_content(symbol)
            if content_analysis:
                esg_data.update(content_analysis)
                esg_data['data_quality_score'] += 0.2
            
            # Get ESG news sentiment
            news_sentiment = await self._get_esg_news_sentiment(symbol)
            if news_sentiment:
                esg_data['esg_news_sentiment'] = news_sentiment
                esg_data['data_quality_score'] += 0.1
            
            # Calculate composite ESG scores
            composite_scores = self._calculate_composite_esg_scores(esg_data)
            esg_data['composite_scores'] = composite_scores
            esg_data['data_quality_score'] += 0.2
            
            # Only return if we have meaningful ESG data
            if esg_data['data_quality_score'] >= 0.3:
                return esg_data
            else:
                self.logger.warning(f"Insufficient ESG data quality for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting ESG data for {symbol}: {e}")
            return None
    
    async def _get_yahoo_esg_data(self, symbol: str) -> Dict[str, Any]:
        """Get ESG data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get sustainability data
            sustainability = ticker.sustainability
            
            if sustainability is not None and not sustainability.empty:
                sustainability_dict = sustainability.to_dict()
                
                return {
                    'esg_scores': sustainability_dict,
                    'environmental_score': self._extract_environmental_score(sustainability_dict),
                    'social_score': self._extract_social_score(sustainability_dict),
                    'governance_score': self._extract_governance_score(sustainability_dict),
                    'overall_esg_score': self._calculate_overall_esg_score(sustainability_dict),
                    'data_source': 'yahoo_finance'
                }
        except Exception as e:
            self.logger.error(f"Yahoo ESG error for {symbol}: {e}")
            
        return {}
    
    async def _scrape_sustainability_reports(self, symbol: str) -> List[Dict]:
        """Scrape publicly available sustainability reports"""
        reports = []
        
        # Get company info to find official website
        try:
            ticker = yf.Ticker(symbol)
            company_info = ticker.info
            website = company_info.get('website', '')
            
            if website:
                potential_urls = self._generate_sustainability_urls(website, symbol)
            else:
                potential_urls = self._generate_fallback_urls(symbol)
                
        except:
            potential_urls = self._generate_fallback_urls(symbol)
        
        # Scrape sustainability content
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in potential_urls:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.text()
                            report_data = self._extract_sustainability_metrics(content, url)
                            if report_data:
                                reports.append({
                                    'url': url,
                                    'data': report_data,
                                    'collection_date': datetime.now().isoformat()
                                })
                            
                            # Limit to first successful report to avoid overloading
                            if reports:
                                break
                                
                except Exception as e:
                    self.logger.debug(f"Failed to scrape {url}: {e}")
                    continue
                    
                await asyncio.sleep(0.5)  # Rate limiting
                
        return reports
    
    def _generate_sustainability_urls(self, website: str, symbol: str) -> List[str]:
        """Generate potential sustainability report URLs"""
        base_domain = urlparse(website).netloc.replace('www.', '')
        
        urls = [
            f"https://www.{base_domain}/sustainability",
            f"https://www.{base_domain}/esg",
            f"https://www.{base_domain}/corporate-responsibility",
            f"https://www.{base_domain}/social-responsibility",
            f"https://www.{base_domain}/environmental",
            f"https://investors.{base_domain}/sustainability",
            f"https://investors.{base_domain}/esg",
            f"https://ir.{base_domain}/sustainability",
            f"https://{base_domain}/about/sustainability",
            f"https://{base_domain}/company/sustainability"
        ]
        
        return urls
    
    def _generate_fallback_urls(self, symbol: str) -> List[str]:
        """Generate fallback URLs when company website is unknown"""
        company_name = symbol.lower()
        
        return [
            f"https://www.{company_name}.com/sustainability",
            f"https://www.{company_name}.com/esg",
            f"https://investors.{company_name}.com/sustainability",
            f"https://ir.{company_name}.com/sustainability"
        ]
    
    def _extract_sustainability_metrics(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract sustainability metrics from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text().lower()
            
            metrics = {
                'url': url,
                'environmental_indicators': {},
                'social_indicators': {},
                'governance_indicators': {},
                'quantitative_data': {},
                'content_analysis': {}
            }
            
            # Environmental indicators
            env_score = 0
            for keyword in self.environmental_keywords:
                if keyword in text:
                    env_score += 1
                    metrics['environmental_indicators'][keyword.replace(' ', '_')] = True
            
            metrics['environmental_indicators']['keyword_count'] = env_score
            metrics['environmental_indicators']['score'] = min(env_score / len(self.environmental_keywords) * 100, 100)
            
            # Social indicators
            social_score = 0
            for keyword in self.social_keywords:
                if keyword in text:
                    social_score += 1
                    metrics['social_indicators'][keyword.replace(' ', '_')] = True
            
            metrics['social_indicators']['keyword_count'] = social_score
            metrics['social_indicators']['score'] = min(social_score / len(self.social_keywords) * 100, 100)
            
            # Governance indicators
            gov_score = 0
            for keyword in self.governance_keywords:
                if keyword in text:
                    gov_score += 1
                    metrics['governance_indicators'][keyword.replace(' ', '_')] = True
            
            metrics['governance_indicators']['keyword_count'] = gov_score
            metrics['governance_indicators']['score'] = min(gov_score / len(self.governance_keywords) * 100, 100)
            
            # Extract quantitative data
            metrics['quantitative_data'] = self._extract_quantitative_esg_data(text)
            
            # Content analysis
            metrics['content_analysis'] = {
                'total_words': len(text.split()),
                'sustainability_focus': (env_score + social_score + gov_score) / len(text.split()) * 1000,
                'content_quality': min((env_score + social_score + gov_score) / 10, 10)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting sustainability metrics from {url}: {e}")
            return {}
    
    def _extract_quantitative_esg_data(self, text: str) -> Dict[str, Any]:
        """Extract quantitative ESG data from text"""
        quantitative_data = {}
        
        # Carbon emissions patterns
        carbon_patterns = [
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million\s+)?(?:tons?|tonnes?)\s+(?:of\s+)?(?:co2|carbon)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)%\s+(?:reduction\s+in\s+)?(?:carbon|emissions)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:mwh|kwh|gwh)\s+(?:renewable|clean)\s+energy'
        ]
        
        for pattern in carbon_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                quantitative_data['carbon_metrics'] = [m.replace(',', '') for m in matches]
        
        # Diversity metrics
        diversity_patterns = [
            r'(\d+(?:\.\d+)?)%\s+(?:women|female|diversity)',
            r'(\d+(?:\.\d+)?)%\s+(?:minority|ethnic|diverse)'
        ]
        
        for pattern in diversity_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                quantitative_data['diversity_metrics'] = matches
        
        # Waste reduction
        waste_patterns = [
            r'(\d+(?:\.\d+)?)%\s+(?:waste\s+)?(?:reduction|recycling)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:tons?|tonnes?)\s+(?:waste\s+)?(?:diverted|recycled)'
        ]
        
        for pattern in waste_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                quantitative_data['waste_metrics'] = [m.replace(',', '') for m in matches]
        
        return quantitative_data
    
    async def _analyze_esg_content(self, symbol: str) -> Dict[str, Any]:
        """Analyze ESG content from various sources"""
        content_analysis = {
            'esg_initiatives': [],
            'sustainability_commitments': [],
            'governance_practices': [],
            'social_programs': []
        }
        
        try:
            # This would integrate with news APIs or content scrapers
            # For now, we'll return a placeholder structure
            content_analysis['analysis_timestamp'] = datetime.now().isoformat()
            content_analysis['data_source'] = 'content_analysis'
            
        except Exception as e:
            self.logger.error(f"Error in ESG content analysis for {symbol}: {e}")
        
        return content_analysis
    
    async def _get_esg_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get ESG-related news sentiment"""
        news_sentiment = {
            'environmental_sentiment': 0.0,
            'social_sentiment': 0.0,
            'governance_sentiment': 0.0,
            'overall_sentiment': 0.0,
            'sentiment_sources': []
        }
        
        try:
            # This would integrate with news sentiment APIs
            # For now, we'll return a placeholder structure
            news_sentiment['analysis_timestamp'] = datetime.now().isoformat()
            news_sentiment['data_source'] = 'news_sentiment'
            
        except Exception as e:
            self.logger.error(f"Error getting ESG news sentiment for {symbol}: {e}")
        
        return news_sentiment
    
    def _extract_environmental_score(self, sustainability_dict: Dict) -> float:
        """Extract environmental score from Yahoo Finance sustainability data"""
        try:
            # Look for environment-related keys in the sustainability data
            env_keys = ['environmentScore', 'environment', 'Environmental']
            for key in env_keys:
                if key in sustainability_dict:
                    value = sustainability_dict[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, dict) and 'value' in value:
                        return float(value['value'])
        except:
            pass
        return 0.0
    
    def _extract_social_score(self, sustainability_dict: Dict) -> float:
        """Extract social score from Yahoo Finance sustainability data"""
        try:
            social_keys = ['socialScore', 'social', 'Social']
            for key in social_keys:
                if key in sustainability_dict:
                    value = sustainability_dict[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, dict) and 'value' in value:
                        return float(value['value'])
        except:
            pass
        return 0.0
    
    def _extract_governance_score(self, sustainability_dict: Dict) -> float:
        """Extract governance score from Yahoo Finance sustainability data"""
        try:
            gov_keys = ['governanceScore', 'governance', 'Governance']
            for key in gov_keys:
                if key in sustainability_dict:
                    value = sustainability_dict[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, dict) and 'value' in value:
                        return float(value['value'])
        except:
            pass
        return 0.0
    
    def _calculate_overall_esg_score(self, sustainability_dict: Dict) -> float:
        """Calculate overall ESG score"""
        try:
            # Look for total ESG score
            total_keys = ['totalEsg', 'totalESG', 'esgScore', 'overallScore']
            for key in total_keys:
                if key in sustainability_dict:
                    value = sustainability_dict[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, dict) and 'value' in value:
                        return float(value['value'])
            
            # Calculate from individual scores
            env_score = self._extract_environmental_score(sustainability_dict)
            social_score = self._extract_social_score(sustainability_dict)
            gov_score = self._extract_governance_score(sustainability_dict)
            
            if env_score > 0 or social_score > 0 or gov_score > 0:
                return (env_score + social_score + gov_score) / 3
                
        except:
            pass
        return 0.0
    
    def _calculate_composite_esg_scores(self, esg_data: Dict) -> Dict[str, float]:
        """Calculate composite ESG scores from all data sources"""
        composite = {
            'environmental_composite': 0.0,
            'social_composite': 0.0,
            'governance_composite': 0.0,
            'overall_composite': 0.0,
            'data_coverage_score': 0.0
        }
        
        try:
            scores = []
            weights = []
            
            # Yahoo Finance ESG scores (weight: 0.4)
            if esg_data.get('environmental_score', 0) > 0:
                scores.append(esg_data['environmental_score'])
                weights.append(0.4)
            
            # Sustainability reports scores (weight: 0.3)
            for report in esg_data.get('sustainability_reports', []):
                report_data = report.get('data', {})
                env_indicators = report_data.get('environmental_indicators', {})
                if env_indicators.get('score', 0) > 0:
                    scores.append(env_indicators['score'])
                    weights.append(0.3)
                    break
            
            # Content analysis scores (weight: 0.3)
            # This would be expanded with actual content analysis
            
            # Calculate weighted composite scores
            if scores and weights:
                composite['environmental_composite'] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                composite['social_composite'] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                composite['governance_composite'] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                composite['overall_composite'] = (
                    composite['environmental_composite'] + 
                    composite['social_composite'] + 
                    composite['governance_composite']
                ) / 3
            
            # Data coverage score
            coverage_factors = [
                1 if esg_data.get('esg_scores') else 0,
                1 if esg_data.get('sustainability_reports') else 0,
                1 if esg_data.get('esg_news_sentiment') else 0
            ]
            composite['data_coverage_score'] = sum(coverage_factors) / len(coverage_factors) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating composite ESG scores: {e}")
        
        return composite
