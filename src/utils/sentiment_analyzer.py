"""
Sentiment Analysis Module for SPR Analyzer

This module analyzes sentiment from news articles, social media,
and company reports to enhance SPR calculations with qualitative insights.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import feedparser

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.text_processor import TextProcessor
from utils.logging_utils import get_logger


@dataclass
class SentimentResult:
    """Results from sentiment analysis"""
    text: str
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: float
    sentiment_label: str
    confidence: float
    source: str
    timestamp: datetime


@dataclass
class QualitativeInsight:
    """Qualitative insight extracted from text analysis"""
    topic: str
    sentiment_trend: str
    key_phrases: List[str]
    impact_score: float
    confidence: float
    supporting_evidence: List[str]
    time_period: Tuple[datetime, datetime]


class SentimentAnalyzer:
    """
    Advanced sentiment analysis for sustainability and financial content
    """
    
    def __init__(self, config_path: str = None):
        """Initialize sentiment analyzer"""
        self.config = ConfigLoader(config_path).config
        self.logger = get_logger(__name__)
        self.text_processor = TextProcessor()
        
        # Initialize sentiment models
        self._load_sentiment_models()
        
        # Initialize NLTK components
        self._setup_nltk()
        
        # Sustainability and financial keywords
        self.sustainability_keywords = self.config['research_processing']['keywords']['sustainability']
        self.financial_keywords = self.config['research_processing']['keywords']['profitability']
        
    def _load_sentiment_models(self):
        """Load pre-trained sentiment analysis models"""
        try:
            self.logger.info("Loading sentiment analysis models...")
            
            # Financial sentiment model
            self.financial_sentiment = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if hasattr(self, 'cuda_available') else -1
            )
            
            # General sentiment model
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if hasattr(self, 'cuda_available') else -1
            )
            
            # ESG-specific sentiment model (if available)
            try:
                self.esg_sentiment = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if hasattr(self, 'cuda_available') else -1
                )
            except:
                self.esg_sentiment = self.general_sentiment
                
            self.logger.info("Sentiment models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment models: {e}")
            # Fallback to simpler models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback sentiment models if advanced models fail"""
        try:
            self.financial_sentiment = pipeline("sentiment-analysis")
            self.general_sentiment = pipeline("sentiment-analysis")
            self.esg_sentiment = pipeline("sentiment-analysis")
            self.logger.info("Loaded fallback sentiment models")
        except Exception as e:
            self.logger.error(f"Error loading fallback models: {e}")
            raise
    
    def _setup_nltk(self):
        """Setup NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("NLTK VADER sentiment analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize VADER: {e}")
            self.vader_analyzer = None
    
    def analyze_text_sentiment(self, text: str, source: str = "unknown", 
                             analysis_type: str = "general") -> SentimentResult:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            source: Source of the text
            analysis_type: Type of analysis ('general', 'financial', 'esg')
            
        Returns:
            SentimentResult object
        """
        try:
            # Clean and preprocess text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Choose appropriate model
            if analysis_type == "financial":
                model = self.financial_sentiment
            elif analysis_type == "esg":
                model = self.esg_sentiment
            else:
                model = self.general_sentiment
            
            # Get sentiment from transformer model
            transformer_result = model(text[:512])  # Limit text length
            
            # Get VADER sentiment scores
            vader_scores = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # Get TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine sentiment scores
            transformer_label = transformer_result[0]['label'].lower()
            transformer_score = transformer_result[0]['score']
            
            # Normalize scores
            if 'positive' in transformer_label or transformer_label == 'pos':
                positive_score = transformer_score
                negative_score = 1 - transformer_score
            elif 'negative' in transformer_label or transformer_label == 'neg':
                positive_score = 1 - transformer_score
                negative_score = transformer_score
            else:
                positive_score = 0.5
                negative_score = 0.5
            
            # Calculate neutral score
            neutral_score = 1 - positive_score - negative_score
            
            # Compound score (weighted average)
            compound_score = (
                0.4 * (positive_score - negative_score) +
                0.3 * vader_scores['compound'] +
                0.3 * textblob_polarity
            )
            
            # Determine final sentiment label
            if compound_score > 0.1:
                sentiment_label = "positive"
                confidence = min(abs(compound_score), 1.0)
            elif compound_score < -0.1:
                sentiment_label = "negative"
                confidence = min(abs(compound_score), 1.0)
            else:
                sentiment_label = "neutral"
                confidence = 1 - abs(compound_score)
            
            return SentimentResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                compound_score=compound_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                source=source,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                positive_score=0.33,
                negative_score=0.33,
                neutral_score=0.34,
                compound_score=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                source=source,
                timestamp=datetime.now()
            )
    
    async def analyze_news_articles(self, company_symbol: str, 
                                  max_articles: int = 50) -> List[SentimentResult]:
        """
        Analyze sentiment of news articles about a company
        
        Args:
            company_symbol: Stock symbol of the company
            max_articles: Maximum number of articles to analyze
            
        Returns:
            List of sentiment results
        """
        self.logger.info(f"Analyzing news sentiment for {company_symbol}")
        
        # Fetch news articles
        articles = await self._fetch_news_articles(company_symbol, max_articles)
        
        # Analyze sentiment of each article
        sentiment_results = []
        for article in articles:
            try:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Determine analysis type based on content
                analysis_type = self._determine_analysis_type(text)
                
                sentiment = self.analyze_text_sentiment(
                    text, 
                    source=f"news_{article.get('source', 'unknown')}", 
                    analysis_type=analysis_type
                )
                sentiment_results.append(sentiment)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing article sentiment: {e}")
                continue
        
        self.logger.info(f"Analyzed sentiment for {len(sentiment_results)} articles")
        return sentiment_results
    
    async def _fetch_news_articles(self, company_symbol: str, 
                                 max_articles: int) -> List[Dict[str, Any]]:
        """Fetch news articles for a company"""
        articles = []
        
        try:
            # Example: Fetch from RSS feeds or news APIs
            # This is a simplified implementation
            
            # Google News RSS (example)
            rss_url = f"https://news.google.com/rss/search?q={company_symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(rss_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:max_articles]:
                            articles.append({
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': 'google_news'
                            })
            
        except Exception as e:
            self.logger.error(f"Error fetching news articles: {e}")
        
        return articles
    
    def _determine_analysis_type(self, text: str) -> str:
        """Determine the appropriate analysis type based on text content"""
        text_lower = text.lower()
        
        # Count financial keywords
        financial_count = sum(1 for keyword in self.financial_keywords 
                            if keyword.lower() in text_lower)
        
        # Count sustainability keywords
        sustainability_count = sum(1 for keyword in self.sustainability_keywords 
                                 if keyword.lower() in text_lower)
        
        if financial_count > sustainability_count:
            return "financial"
        elif sustainability_count > 0:
            return "esg"
        else:
            return "general"
    
    def analyze_social_media_sentiment(self, company_symbol: str, 
                                     platform: str = "twitter") -> List[SentimentResult]:
        """
        Analyze sentiment from social media posts
        
        Args:
            company_symbol: Stock symbol of the company
            platform: Social media platform
            
        Returns:
            List of sentiment results
        """
        # This is a placeholder implementation
        # In practice, you would integrate with social media APIs
        self.logger.info(f"Social media sentiment analysis for {company_symbol} on {platform}")
        
        # Mock data for demonstration
        mock_posts = [
            f"{company_symbol} is leading in sustainable practices! Great ESG performance.",
            f"Disappointed with {company_symbol}'s environmental impact this quarter.",
            f"{company_symbol} quarterly results looking strong, sustainability initiatives paying off.",
        ]
        
        results = []
        for post in mock_posts:
            sentiment = self.analyze_text_sentiment(post, source=f"{platform}_post", analysis_type="esg")
            results.append(sentiment)
        
        return results
    
    def extract_qualitative_insights(self, sentiment_results: List[SentimentResult], 
                                   time_window_days: int = 30) -> List[QualitativeInsight]:
        """
        Extract high-level qualitative insights from sentiment analysis results
        
        Args:
            sentiment_results: List of sentiment analysis results
            time_window_days: Time window for analysis in days
            
        Returns:
            List of qualitative insights
        """
        insights = []
        
        if not sentiment_results:
            return insights
        
        # Group results by time periods
        end_date = max(r.timestamp for r in sentiment_results)
        start_date = end_date - timedelta(days=time_window_days)
        
        # Filter results within time window
        recent_results = [r for r in sentiment_results 
                         if start_date <= r.timestamp <= end_date]
        
        if not recent_results:
            return insights
        
        # Calculate overall sentiment trends
        avg_positive = np.mean([r.positive_score for r in recent_results])
        avg_negative = np.mean([r.negative_score for r in recent_results])
        avg_compound = np.mean([r.compound_score for r in recent_results])
        
        # Determine sentiment trend
        if avg_compound > 0.1:
            sentiment_trend = "positive"
        elif avg_compound < -0.1:
            sentiment_trend = "negative"
        else:
            sentiment_trend = "neutral"
        
        # Extract key phrases
        all_texts = [r.text for r in recent_results]
        key_phrases = self._extract_key_phrases(all_texts)
        
        # Calculate impact score
        impact_score = min(abs(avg_compound) * len(recent_results) / 10, 1.0)
        
        # Calculate confidence
        confidence = np.mean([r.confidence for r in recent_results])
        
        # Create insight
        insight = QualitativeInsight(
            topic="Overall Sentiment",
            sentiment_trend=sentiment_trend,
            key_phrases=key_phrases[:10],  # Top 10 phrases
            impact_score=impact_score,
            confidence=confidence,
            supporting_evidence=[r.text for r in recent_results[:3]],  # Top 3 examples
            time_period=(start_date, end_date)
        )
        
        insights.append(insight)
        
        # Topic-specific insights
        topics = ["sustainability", "financial_performance", "esg"]
        for topic in topics:
            topic_insight = self._analyze_topic_sentiment(recent_results, topic)
            if topic_insight:
                insights.append(topic_insight)
        
        return insights
    
    def _extract_key_phrases(self, texts: List[str], max_phrases: int = 20) -> List[str]:
        """Extract key phrases from a collection of texts"""
        try:
            # Combine all texts
            combined_text = " ".join(texts)
            
            # Use text processor to extract keywords
            keywords = self.text_processor.extract_keywords(combined_text, num_keywords=max_phrases)
            
            return keywords
            
        except Exception as e:
            self.logger.warning(f"Error extracting key phrases: {e}")
            return []
    
    def _analyze_topic_sentiment(self, sentiment_results: List[SentimentResult], 
                               topic: str) -> Optional[QualitativeInsight]:
        """Analyze sentiment for a specific topic"""
        try:
            # Filter results relevant to topic
            topic_keywords = {
                "sustainability": ["sustainable", "green", "environment", "carbon", "renewable"],
                "financial_performance": ["profit", "revenue", "earnings", "financial", "performance"],
                "esg": ["esg", "governance", "social", "responsibility", "ethics"]
            }
            
            keywords = topic_keywords.get(topic, [])
            if not keywords:
                return None
            
            # Find relevant results
            relevant_results = []
            for result in sentiment_results:
                text_lower = result.text.lower()
                if any(keyword in text_lower for keyword in keywords):
                    relevant_results.append(result)
            
            if len(relevant_results) < 3:  # Need minimum results for meaningful insight
                return None
            
            # Calculate topic-specific metrics
            avg_compound = np.mean([r.compound_score for r in relevant_results])
            confidence = np.mean([r.confidence for r in relevant_results])
            
            # Determine trend
            if avg_compound > 0.1:
                sentiment_trend = "positive"
            elif avg_compound < -0.1:
                sentiment_trend = "negative"
            else:
                sentiment_trend = "neutral"
            
            # Extract key phrases
            topic_texts = [r.text for r in relevant_results]
            key_phrases = self._extract_key_phrases(topic_texts, max_phrases=5)
            
            return QualitativeInsight(
                topic=topic.replace("_", " ").title(),
                sentiment_trend=sentiment_trend,
                key_phrases=key_phrases,
                impact_score=min(abs(avg_compound) * len(relevant_results) / 5, 1.0),
                confidence=confidence,
                supporting_evidence=[r.text for r in relevant_results[:2]],
                time_period=(
                    min(r.timestamp for r in relevant_results),
                    max(r.timestamp for r in relevant_results)
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing topic sentiment: {e}")
            return None
    
    def calculate_sentiment_impact_score(self, sentiment_results: List[SentimentResult]) -> float:
        """
        Calculate an overall sentiment impact score for SPR analysis
        
        Args:
            sentiment_results: List of sentiment results
            
        Returns:
            Impact score between -1 and 1
        """
        if not sentiment_results:
            return 0.0
        
        # Weight by confidence and recency
        weighted_scores = []
        now = datetime.now()
        
        for result in sentiment_results:
            # Recency weight (more recent = higher weight)
            days_old = (now - result.timestamp).days
            recency_weight = max(0.1, 1.0 - (days_old / 30))  # Decay over 30 days
            
            # Final weighted score
            weighted_score = result.compound_score * result.confidence * recency_weight
            weighted_scores.append(weighted_score)
        
        # Calculate weighted average
        if weighted_scores:
            impact_score = np.mean(weighted_scores)
            return max(-1.0, min(1.0, impact_score))  # Clamp to [-1, 1]
        
        return 0.0


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Example: Analyze news sentiment
        news_sentiment = await analyzer.analyze_news_articles("TSLA", max_articles=10)
        
        print(f"Analyzed {len(news_sentiment)} news articles")
        for sentiment in news_sentiment[:3]:
            print(f"Text: {sentiment.text}")
            print(f"Sentiment: {sentiment.sentiment_label} (confidence: {sentiment.confidence:.2f})")
            print(f"Compound score: {sentiment.compound_score:.2f}")
            print("-" * 50)
        
        # Extract insights
        insights = analyzer.extract_qualitative_insights(news_sentiment)
        print(f"\nExtracted {len(insights)} insights:")
        for insight in insights:
            print(f"Topic: {insight.topic}")
            print(f"Trend: {insight.sentiment_trend}")
            print(f"Impact score: {insight.impact_score:.2f}")
            print(f"Key phrases: {', '.join(insight.key_phrases[:5])}")
            print("-" * 30)
        
        # Calculate overall impact score
        impact_score = analyzer.calculate_sentiment_impact_score(news_sentiment)
        print(f"\nOverall sentiment impact score: {impact_score:.2f}")
    
    # Run the example
    asyncio.run(main())
