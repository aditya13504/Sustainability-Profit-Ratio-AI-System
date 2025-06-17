"""
Google Gemini 2.5 Pro API Integration for SPR Analyzer
Provides AI-powered analysis, summarization, and insights for financial and sustainability data
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("⚠️ Google Generative AI library not installed. Run: pip install google-generativeai")
    genai = None

# Add src to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader


class GeminiAIAssistant:
    """
    Google Gemini 2.5 Pro AI Assistant for SPR Analysis
    Provides advanced AI capabilities for financial and sustainability analysis
    """
    
    def __init__(self, config_loader: ConfigLoader = None):
        """Initialize Gemini AI Assistant"""
        if genai is None:
            raise ImportError("Google Generative AI library is required. Install with: pip install google-generativeai")
        
        self.config = config_loader or ConfigLoader()
        self.logger = logging.getLogger(__name__)
        
        # Get API key from config or environment
        self.api_key = (
            self.config.get('api_keys.google_gemini_api_key') or 
            self.config.get('ai.gemini.api_key') or
            self.config.get('apis.research.google_gemini.api_key') or
            os.getenv('GOOGLE_GEMINI_API_KEY')
        )
        
        if not self.api_key:
            raise ValueError("Google Gemini API key not found in config or environment variables")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Get model configuration
        self.model_name = self.config.get('ai.gemini.model', 'gemini-1.5-flash')
        self.max_tokens = self.config.get('ai.gemini.max_tokens', 8192)
        self.temperature = self.config.get('ai.gemini.temperature', 0.7)
        
        # Rate limiting configuration
        self.requests_per_minute = self.config.get('ai.gemini.rate_limiting.requests_per_minute', 8)
        self.retry_attempts = self.config.get('ai.gemini.rate_limiting.retry_attempts', 3)
        self.backoff_multiplier = self.config.get('ai.gemini.rate_limiting.backoff_multiplier', 2)
        self.max_delay = self.config.get('ai.gemini.rate_limiting.max_delay', 60)
        
        # Request tracking for rate limiting
        self.request_times = []
        self.last_request_time = 0
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.config.get('ai.gemini.top_p', 0.95),
            "top_k": self.config.get('ai.gemini.top_k', 40),
            "max_output_tokens": self.max_tokens,
            "response_mime_type": "text/plain"
        }
        
        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )        
        self.logger.info(f"Gemini AI Assistant initialized with model: {self.model_name}")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid API quota exceeded errors"""
        current_time = time.time()
        
        # Clean old request times (older than 1 minute)
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                # Clean request times again after waiting
                current_time = time.time()
                minute_ago = current_time - 60
                self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Record this request
        self.request_times.append(current_time)
        self.last_request_time = current_time
    
    def _make_request_with_retry(self, prompt: str, max_retries: int = None) -> str:
        """Make a request to Gemini with retry logic and rate limiting"""
        max_retries = max_retries or self.retry_attempts
        
        for attempt in range(max_retries + 1):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                # Make the request
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini API")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    if attempt < max_retries:
                        wait_time = min(self.backoff_multiplier ** attempt * 5, self.max_delay)
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                
                # For other errors, log and raise
                self.logger.error(f"Error making request to Gemini (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise e        
        raise Exception(f"Failed to get response after {max_retries + 1} attempts")
    
    def analyze_company_data(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze company data using Gemini AI
        
        Args:
            company_data: Dictionary containing company financial and sustainability data
            
        Returns:
            AI analysis with insights, recommendations, and risk assessment
        """
        try:
            prompt = self._create_company_analysis_prompt(company_data)
            response_text = self._make_request_with_retry(prompt)
              # Parse response
            analysis = self._parse_analysis_response(response_text)
            
            return {
                'company': company_data.get('name', 'Unknown'),
                'symbol': company_data.get('symbol', 'N/A'),
                'analysis_date': datetime.now().isoformat(),
                'gemini_analysis': analysis,
                'raw_response': response_text,
                'model_used': self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing company data: {e}")
            return {
                'error': str(e),
                'company': company_data.get('name', 'Unknown'),
                'analysis_date': datetime.now().isoformat()
            }
    
    def summarize_research_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize research papers using Gemini AI
        
        Args:
            papers: List of research paper data
              Returns:
            AI-generated summary with key insights and trends
        """
        try:
            prompt = self._create_research_summary_prompt(papers)
            response_text = self._make_request_with_retry(prompt)
            
            return {
                'summary_date': datetime.now().isoformat(),
                'papers_analyzed': len(papers),
                'summary': response_text,
                'model_used': self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing research papers: {e}")
            return {
                'error': str(e),
                'papers_analyzed': len(papers),
                'summary_date': datetime.now().isoformat()
            }
    
    def analyze_market_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market sentiment from news data
        
        Args:
            news_data: List of news articles and data
              Returns:
            Sentiment analysis with market insights
        """
        try:
            prompt = self._create_sentiment_analysis_prompt(news_data)
            response_text = self._make_request_with_retry(prompt)
            
            return {
                'analysis_date': datetime.now().isoformat(),
                'articles_analyzed': len(news_data),
                'sentiment_analysis': response_text,
                'model_used': self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {
                'error': str(e),
                'articles_analyzed': len(news_data),
                'analysis_date': datetime.now().isoformat()
            }
    
    def generate_spr_insights(self, spr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights on Sustainability Profit Ratio (SPR) data
        
        Args:
            spr_data: SPR calculation results and related data
              Returns:
            AI-generated insights and recommendations
        """
        try:
            prompt = self._create_spr_insights_prompt(spr_data)
            response_text = self._make_request_with_retry(prompt)
            
            return {
                'insights_date': datetime.now().isoformat(),
                'spr_insights': response_text,
                'model_used': self.model_name,
                'data_analyzed': spr_data
            }
            
        except Exception as e:
            self.logger.error(f"Error generating SPR insights: {e}")
            return {
                'error': str(e),
                'insights_date': datetime.now().isoformat()
            }
    
    def enhance_company_registry_data(self, registry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance company registry data with AI-generated insights
        
        Args:
            registry_data: List of company registry entries
              Returns:
            Enhanced data with AI insights
        """
        enhanced_data = []
        
        for company in registry_data:
            try:
                prompt = self._create_company_enhancement_prompt(company)
                response_text = self._make_request_with_retry(prompt)
                
                # Add AI enhancement to company data
                enhanced_company = company.copy()
                enhanced_company['ai_enhancement'] = {
                    'analysis': response_text,
                    'enhanced_date': datetime.now().isoformat(),
                    'model_used': self.model_name
                }
                
                enhanced_data.append(enhanced_company)
                
            except Exception as e:
                self.logger.error(f"Error enhancing company data for {company.get('name', 'Unknown')}: {e}")
                enhanced_data.append(company)  # Add original data if enhancement fails
        
        return enhanced_data
    
    def _create_company_analysis_prompt(self, company_data: Dict[str, Any]) -> str:
        """Create prompt for company analysis"""
        prompt = f"""
Analyze the following company data and provide comprehensive insights:

Company: {company_data.get('name', 'Unknown')}
Symbol: {company_data.get('symbol', 'N/A')}
Industry: {company_data.get('industry', 'N/A')}
Market Cap: {company_data.get('market_cap', 'N/A')}

Financial Data:
{json.dumps(company_data.get('financial_data', {}), indent=2)}

Sustainability Data:
{json.dumps(company_data.get('sustainability_data', {}), indent=2)}

Please provide analysis in the following structure:
1. Financial Performance Assessment
2. Sustainability Impact Evaluation
3. Risk Assessment
4. Growth Potential
5. Investment Recommendation
6. Key Strengths and Weaknesses
7. Future Outlook

Focus on actionable insights and data-driven conclusions.
"""
        return prompt
    
    def _create_research_summary_prompt(self, papers: List[Dict[str, Any]]) -> str:
        """Create prompt for research paper summarization"""
        papers_text = ""
        for i, paper in enumerate(papers[:10], 1):  # Limit to 10 papers to avoid token limits
            papers_text += f"""
Paper {i}:
Title: {paper.get('title', 'Unknown')}
Authors: {paper.get('authors', 'Unknown')}
Abstract: {paper.get('abstract', 'No abstract available')}
Keywords: {paper.get('keywords', [])}
---
"""
        
        prompt = f"""
Analyze the following research papers and provide a comprehensive summary:

{papers_text}

Please provide a summary that includes:
1. Key Research Trends
2. Major Findings and Insights
3. Methodological Approaches
4. Implications for Investment and Sustainability
5. Future Research Directions
6. Practical Applications

Focus on actionable insights for financial and sustainability analysis.
"""
        return prompt
    
    def _create_sentiment_analysis_prompt(self, news_data: List[Dict[str, Any]]) -> str:
        """Create prompt for sentiment analysis"""
        news_text = ""
        for i, article in enumerate(news_data[:15], 1):  # Limit to 15 articles
            news_text += f"""
Article {i}:
Title: {article.get('title', 'Unknown')}
Source: {article.get('source', 'Unknown')}
Published: {article.get('published_date', 'Unknown')}
Content: {article.get('content', article.get('description', 'No content available'))[:500]}...
---
"""
        
        prompt = f"""
Analyze the sentiment and market implications of the following news articles:

{news_text}

Please provide analysis including:
1. Overall Market Sentiment (Positive/Negative/Neutral)
2. Key Market Drivers and Concerns
3. Sector-Specific Trends
4. Risk Factors Identified
5. Investment Implications
6. Short-term vs Long-term Outlook

Provide specific, actionable insights for investment decisions.
"""
        return prompt
    
    def _create_spr_insights_prompt(self, spr_data: Dict[str, Any]) -> str:
        """Create prompt for SPR insights"""
        prompt = f"""
Analyze the following Sustainability Profit Ratio (SPR) data and provide insights:

SPR Data:
{json.dumps(spr_data, indent=2)}

Please provide analysis including:
1. SPR Score Interpretation
2. Sustainability vs Profitability Balance
3. Benchmark Comparison
4. Improvement Recommendations
5. Investment Attractiveness
6. ESG Risk Assessment
7. Long-term Value Proposition

Focus on actionable recommendations for sustainable investing.
"""
        return prompt
    
    def _create_company_enhancement_prompt(self, company: Dict[str, Any]) -> str:
        """Create prompt for company data enhancement"""
        prompt = f"""
Enhance the following company registry data with insights and analysis:

Company: {company.get('name', 'Unknown')}
Registration Number: {company.get('registration_number', 'N/A')}
Status: {company.get('status', 'N/A')}
Industry: {company.get('industry', 'N/A')}
Location: {company.get('location', 'N/A')}

Raw Data:
{json.dumps(company, indent=2)}

Please provide brief insights on:
1. Business Model Assessment
2. Market Position
3. Growth Potential
4. Risk Factors
5. Industry Analysis

Keep response concise (under 300 words) and focus on key insights.
"""
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract structured information from the response
            sections = {
                'financial_assessment': '',
                'sustainability_evaluation': '',
                'risk_assessment': '',
                'growth_potential': '',
                'recommendation': '',
                'strengths_weaknesses': '',
                'outlook': '',
                'full_analysis': response_text
            }
            
            # Simple parsing - can be enhanced with more sophisticated extraction
            lines = response_text.split('\n')
            current_section = 'full_analysis'
            
            for line in lines:
                line = line.strip()
                if 'financial performance' in line.lower():
                    current_section = 'financial_assessment'
                elif 'sustainability' in line.lower() and 'impact' in line.lower():
                    current_section = 'sustainability_evaluation'
                elif 'risk' in line.lower():
                    current_section = 'risk_assessment'
                elif 'growth' in line.lower():
                    current_section = 'growth_potential'
                elif 'recommendation' in line.lower():
                    current_section = 'recommendation'
                elif 'strength' in line.lower() or 'weakness' in line.lower():
                    current_section = 'strengths_weaknesses'
                elif 'outlook' in line.lower() or 'future' in line.lower():
                    current_section = 'outlook'
                
                if current_section != 'full_analysis' and line:
                    sections[current_section] += line + ' '
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return {'full_analysis': response_text, 'parsing_error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Gemini model"""
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'api_configured': bool(self.api_key),
            'initialization_time': datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    """Test the Gemini AI Assistant"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Gemini AI Assistant')
    parser.add_argument('--test-config', action='store_true', help='Test configuration and model info')
    parser.add_argument('--test-analysis', action='store_true', help='Test company analysis')
    parser.add_argument('--test-summary', action='store_true', help='Test research summarization')
    
    args = parser.parse_args()
    
    try:
        print("🤖 Initializing Gemini AI Assistant...")
        assistant = GeminiAIAssistant()
        print("✅ Gemini AI Assistant initialized successfully!")
        
        if args.test_config:
            print("\n📊 Model Information:")
            model_info = assistant.get_model_info()
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        
        if args.test_analysis:
            print("\n🏢 Testing company analysis...")
            sample_company = {
                'name': 'Tesla Inc.',
                'symbol': 'TSLA',
                'industry': 'Electric Vehicles',
                'market_cap': '800B',
                'financial_data': {
                    'revenue': '96.8B',
                    'profit_margin': '7.3%',
                    'roi': '12.5%'
                },
                'sustainability_data': {
                    'carbon_footprint': 'Low',
                    'renewable_energy': 'High',
                    'esg_score': '8.2/10'
                }
            }
            
            analysis = assistant.analyze_company_data(sample_company)
            print(f"✅ Analysis completed for {sample_company['name']}")
            print(f"Analysis preview: {analysis.get('gemini_analysis', {}).get('full_analysis', 'N/A')[:200]}...")
        
        if args.test_summary:
            print("\n📚 Testing research summarization...")
            sample_papers = [
                {
                    'title': 'Sustainable Finance and Green Investments',
                    'authors': 'Smith, J. et al.',
                    'abstract': 'This paper examines the relationship between sustainable finance practices and long-term returns...',
                    'keywords': ['sustainability', 'finance', 'ESG']
                },
                {
                    'title': 'Corporate Sustainability and Financial Performance',
                    'authors': 'Johnson, A. et al.',
                    'abstract': 'Analysis of how corporate sustainability initiatives impact financial performance metrics...',
                    'keywords': ['corporate sustainability', 'financial performance', 'ROI']
                }
            ]
            
            summary = assistant.summarize_research_papers(sample_papers)
            print(f"✅ Research summary completed for {summary['papers_analyzed']} papers")
            print(f"Summary preview: {summary.get('summary', 'N/A')[:200]}...")
        
        if not any([args.test_config, args.test_analysis, args.test_summary]):
            print("\n❓ No specific test requested. Use --help for options.")
            print("Available tests:")
            print("  --test-config    Test configuration and model info")
            print("  --test-analysis  Test company analysis")
            print("  --test-summary   Test research summarization")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
