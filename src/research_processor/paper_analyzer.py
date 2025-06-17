"""
Research Paper Analyzer for Sustainability and Profitability Insights

This module processes academic research papers to extract insights about
sustainability practices and their correlation with financial performance.
"""

import os
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import requests
from bs4 import BeautifulSoup
import arxiv
import feedparser
from scholarly import scholarly

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.text_processor import TextProcessor


@dataclass
class ResearchPaper:
    """Data class for research paper information"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    source: str
    url: str
    citations: int = 0
    sustainability_score: float = 0.0
    profitability_relevance: float = 0.0
    key_insights: List[str] = None
    

@dataclass
class SustainabilityInsight:
    """Data class for extracted sustainability insights"""
    practice: str
    impact_description: str
    financial_correlation: float
    confidence_score: float
    supporting_evidence: List[str]
    source_papers: List[str]


class ResearchAnalyzer:
    """
    Analyzes research papers to extract sustainability and profitability insights
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the research analyzer with configuration"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize NLP models
        self._load_models()
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Cache for processed papers
        self.paper_cache = {}
        
        # Sustainability keywords
        self.sustainability_keywords = self.config['research_processing']['keywords']['sustainability']
        self.profitability_keywords = self.config['research_processing']['keywords']['profitability']
        
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
        
    def _load_models(self):
        """Load pre-trained NLP models"""
        try:
            self.logger.info("Loading NLP models...")
            
            # Summarization model
            model_name = self.config['models']['nlp']['summarization']['model_name']
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentiment analysis model
            sentiment_model = self.config['models']['nlp']['sentiment']['model_name']
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Embedding model for similarity
            embedding_model = self.config['models']['nlp']['embedding']['model_name']
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.embedding_model = AutoModel.from_pretrained(embedding_model)
            
            self.logger.info("NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
            
    async def search_papers(self, query: str, max_papers: int = 50) -> List[ResearchPaper]:
        """
        Search for research papers from multiple sources
        
        Args:
            query: Search query string
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            List of ResearchPaper objects
        """
        papers = []
        
        # Search from multiple sources concurrently
        tasks = []
        
        if self.config['apis']['research']['arxiv']['enabled']:
            tasks.append(self._search_arxiv(query, max_papers // 3))
            
        if self.config['apis']['research']['google_scholar']['enabled']:
            tasks.append(self._search_google_scholar(query, max_papers // 3))
            
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                papers.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Search failed: {result}")
                
        # Remove duplicates and sort by relevance
        papers = self._deduplicate_papers(papers)
        papers = papers[:max_papers]
        
        self.logger.info(f"Found {len(papers)} papers for query: {query}")
        return papers
        
    async def _search_arxiv(self, query: str, max_papers: int) -> List[ResearchPaper]:
        """Search arXiv for research papers"""
        papers = []
        
        try:
            # Create search query
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                paper = ResearchPaper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    publication_date=result.published,
                    source="arXiv",
                    url=result.pdf_url,
                    citations=0  # arXiv doesn't provide citation count
                )
                papers.append(paper)
                
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            
        return papers
        
    async def _search_google_scholar(self, query: str, max_papers: int) -> List[ResearchPaper]:
        """Search Google Scholar for research papers"""
        papers = []
        
        try:
            # Note: This is a simplified implementation
            # In production, you might want to use a more robust Google Scholar API
            search_query = scholarly.search_pubs(query)
            
            count = 0
            for pub in search_query:
                if count >= max_papers:
                    break
                    
                try:
                    paper = ResearchPaper(
                        title=pub.get('title', ''),
                        authors=pub.get('author', []),
                        abstract=pub.get('abstract', ''),
                        publication_date=self._parse_date(pub.get('year', '')),
                        source="Google Scholar",
                        url=pub.get('url', ''),
                        citations=pub.get('num_citations', 0)
                    )
                    papers.append(paper)
                    count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing paper: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching Google Scholar: {e}")
            
        return papers
        
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            if isinstance(date_str, int):
                return datetime(date_str, 1, 1)
            elif isinstance(date_str, str):
                return datetime(int(date_str), 1, 1)
            else:
                return datetime.now()
        except:
            return datetime.now()
            
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        if not papers:
            return papers
            
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Simple deduplication based on title
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
                
        return unique_papers
        
    def analyze_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """
        Analyze papers to extract sustainability and profitability insights
        
        Args:
            papers: List of research papers to analyze
            
        Returns:
            List of analyzed papers with scores and insights
        """
        analyzed_papers = []
        
        for paper in papers:
            try:
                # Analyze sustainability relevance
                sustainability_score = self._calculate_sustainability_score(paper)
                
                # Analyze profitability relevance
                profitability_relevance = self._calculate_profitability_relevance(paper)
                
                # Extract key insights
                key_insights = self._extract_key_insights(paper)
                
                # Update paper with analysis results
                paper.sustainability_score = sustainability_score
                paper.profitability_relevance = profitability_relevance
                paper.key_insights = key_insights
                
                analyzed_papers.append(paper)
                
            except Exception as e:
                self.logger.warning(f"Error analyzing paper '{paper.title}': {e}")
                continue
                
        # Sort by combined relevance score
        analyzed_papers.sort(
            key=lambda p: (p.sustainability_score + p.profitability_relevance) / 2,
            reverse=True
        )
        
        self.logger.info(f"Analyzed {len(analyzed_papers)} papers")
        return analyzed_papers
        
    def _calculate_sustainability_score(self, paper: ResearchPaper) -> float:
        """Calculate sustainability relevance score for a paper"""
        text = f"{paper.title} {paper.abstract}".lower()
        
        # Count sustainability keyword matches
        keyword_matches = sum(1 for keyword in self.sustainability_keywords 
                            if keyword.lower() in text)
        
        # Normalize score
        max_possible_matches = len(self.sustainability_keywords)
        keyword_score = keyword_matches / max_possible_matches if max_possible_matches > 0 else 0
          # Use sentiment analysis to boost positive sustainability content
        try:
            sentiment_result = self.sentiment_analyzer(paper.abstract[:512])
            sentiment_boost = 0.1 if sentiment_result[0]['label'] == 'POSITIVE' else 0
        except:
            sentiment_boost = 0
            
        return min(keyword_score + sentiment_boost, 1.0)
    
    def _calculate_profitability_relevance(self, paper: ResearchPaper) -> float:
        """Calculate profitability relevance score for a paper"""
        text = f"{paper.title} {paper.abstract}".lower()
        
        # Extended profitability-related keywords
        extended_keywords = self.profitability_keywords + [
            'financial returns', 'economic performance', 'business value', 
            'financial benefits', 'economic impact', 'cost savings',
            'revenue', 'profit', 'earnings', 'margins', 'returns',
            'financial success', 'economic value', 'business performance',
            'competitive advantage', 'market performance', 'shareholder value'
        ]
        
        # Count profitability keyword matches
        keyword_matches = sum(1 for keyword in extended_keywords 
                            if keyword.lower() in text)
        
        # Normalize score but give minimum score if any financial terms found
        max_possible_matches = len(extended_keywords)
        raw_score = keyword_matches / max_possible_matches if max_possible_matches > 0 else 0
        
        # Ensure minimum relevance if paper mentions sustainability and any financial terms
        sustainability_mentioned = any(keyword.lower() in text for keyword in self.sustainability_keywords)
        financial_mentioned = keyword_matches > 0
        
        if sustainability_mentioned and financial_mentioned:
            return max(raw_score, 0.3)  # Minimum 30% relevance
        elif financial_mentioned:
            return max(raw_score, 0.2)  # Minimum 20% relevance
        else:
            return raw_score
        
    def _extract_key_insights(self, paper: ResearchPaper) -> List[str]:
        """Extract key insights from paper using summarization"""
        try:
            # Summarize the abstract to get key insights
            if len(paper.abstract) > 50:
                summary = self.summarizer(
                    paper.abstract,
                    max_length=self.config['models']['nlp']['summarization']['max_length'],
                    min_length=self.config['models']['nlp']['summarization']['min_length'],
                    do_sample=False
                )
                
                # Split summary into key points
                summary_text = summary[0]['summary_text']
                insights = sent_tokenize(summary_text)
                
                return [insight.strip() for insight in insights if len(insight.strip()) > 20]
            else:
                return [paper.abstract]
                
        except Exception as e:
            self.logger.warning(f"Error extracting insights: {e}")
            return [paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract]
            
    def extract_sustainability_insights(self, papers: List[ResearchPaper]) -> List[SustainabilityInsight]:
        """
        Extract structured sustainability insights from analyzed papers
        
        Args:
            papers: List of analyzed research papers
            
        Returns:
            List of sustainability insights
        """
        insights = []
        
        # Group papers by sustainability practices
        practice_groups = self._group_by_practices(papers)
        
        for practice, practice_papers in practice_groups.items():
            # Aggregate insights for this practice
            insight = self._aggregate_practice_insights(practice, practice_papers)
            if insight:
                insights.append(insight)
                  # Sort by confidence score
        insights.sort(key=lambda i: i.confidence_score, reverse=True)
        
        self.logger.info(f"Extracted {len(insights)} sustainability insights")
        return insights
    
    def _group_by_practices(self, papers: List[ResearchPaper]) -> Dict[str, List[ResearchPaper]]:
        """Group papers by sustainability practices"""
        practice_groups = {}
        
        practice_keywords = {
            "renewable_energy": ["renewable energy", "solar", "wind", "clean energy"],
            "carbon_reduction": ["carbon", "emissions", "CO2", "greenhouse gas"],
            "resource_efficiency": ["efficiency", "waste reduction", "recycling"],
            "sustainable_supply_chain": ["supply chain", "sustainable sourcing"],
            "green_technology": ["green technology", "cleantech", "environmental technology"],
            "esg_governance": ["esg", "environmental social governance", "corporate governance", "social responsibility"],
            "sustainability_strategy": ["sustainability", "sustainable", "green", "environmental"],
            "digital_transformation": ["digital transformation", "digitalization", "technology adoption"],
            "financial_sustainability": ["sustainable finance", "green finance", "impact investing"]
        }
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            paper_assigned = False
            
            for practice, keywords in practice_keywords.items():
                if any(keyword in text for keyword in keywords):
                    if practice not in practice_groups:
                        practice_groups[practice] = []
                    practice_groups[practice].append(paper)
                    paper_assigned = True
            
            # If paper wasn't assigned to any specific practice but has sustainability score > 0,
            # assign it to general sustainability
            if not paper_assigned and paper.sustainability_score > 0:
                if "general_sustainability" not in practice_groups:
                    practice_groups["general_sustainability"] = []
                practice_groups["general_sustainability"].append(paper)
                    
        return practice_groups
        
    def _aggregate_practice_insights(self, practice: str, papers: List[ResearchPaper]) -> Optional[SustainabilityInsight]:
        """Aggregate insights for a specific sustainability practice"""
        if not papers:
            return None
            
        # Calculate average financial correlation
        financial_correlations = [p.profitability_relevance for p in papers if p.profitability_relevance > 0]
        avg_financial_correlation = np.mean(financial_correlations) if financial_correlations else 0
        
        # Calculate confidence based on number of papers and citation counts
        total_citations = sum(p.citations for p in papers)
        confidence_score = min((len(papers) * 0.1 + total_citations * 0.001), 1.0)
        
        # Extract supporting evidence
        supporting_evidence = []
        for paper in papers[:3]:  # Top 3 papers
            if paper.key_insights:
                supporting_evidence.extend(paper.key_insights[:2])
                
        # Create impact description
        impact_description = self._generate_impact_description(practice, papers)
        
        return SustainabilityInsight(
            practice=practice.replace("_", " ").title(),
            impact_description=impact_description,
            financial_correlation=avg_financial_correlation,
            confidence_score=confidence_score,
            supporting_evidence=supporting_evidence[:5],  # Limit to 5 pieces of evidence
            source_papers=[p.title for p in papers[:5]]  # Reference top 5 papers
        )
        
    def _generate_impact_description(self, practice: str, papers: List[ResearchPaper]) -> str:
        """Generate a description of the practice's impact"""
        # This is a simplified version - in production, you might use a more sophisticated NLG approach
        high_correlation_papers = [p for p in papers if p.profitability_relevance > 0.5]
        
        if len(high_correlation_papers) > len(papers) * 0.6:
            return f"{practice.replace('_', ' ').title()} shows strong positive correlation with financial performance based on research evidence."
        elif len(high_correlation_papers) > len(papers) * 0.3:
            return f"{practice.replace('_', ' ').title()} shows moderate correlation with financial performance."
        else:
            return f"{practice.replace('_', ' ').title()} shows limited evidence of direct financial impact."
            
    async def analyze_sustainability_papers(self, query: str, max_papers: int = 50) -> Dict[str, Any]:
        """
        Main method to analyze sustainability papers for a given query
        
        Args:
            query: Search query for sustainability research
            max_papers: Maximum number of papers to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Starting analysis for query: {query}")
        
        # Search for papers
        papers = await self.search_papers(query, max_papers)
        
        if not papers:
            self.logger.warning("No papers found for the query")
            return {"papers": [], "insights": [], "summary": "No papers found"}
            
        # Analyze papers
        analyzed_papers = self.analyze_papers(papers)
        
        # Extract sustainability insights
        insights = self.extract_sustainability_insights(analyzed_papers)
        
        # Generate summary
        summary = self._generate_analysis_summary(analyzed_papers, insights)
        
        result = {
            "papers": analyzed_papers,
            "insights": insights,
            "summary": summary,
            "metadata": {
                "query": query,
                "total_papers": len(analyzed_papers),
                "analysis_date": datetime.now(),
                "avg_sustainability_score": np.mean([p.sustainability_score for p in analyzed_papers]),
                "avg_profitability_relevance": np.mean([p.profitability_relevance for p in analyzed_papers])
            }
        }
        
        self.logger.info("Analysis completed successfully")
        return result
        
    def _generate_analysis_summary(self, papers: List[ResearchPaper], insights: List[SustainabilityInsight]) -> str:
        """Generate a summary of the analysis results"""
        if not papers:
            return "No papers were analyzed."
            
        high_relevance_papers = [p for p in papers if p.sustainability_score > 0.7 and p.profitability_relevance > 0.5]
        
        summary_parts = [
            f"Analyzed {len(papers)} research papers.",
            f"Found {len(high_relevance_papers)} papers with high sustainability and profitability relevance.",
            f"Identified {len(insights)} key sustainability practices with financial correlations."
        ]
        
        if insights:
            top_insight = insights[0]
            summary_parts.append(f"Top insight: {top_insight.practice} shows {top_insight.financial_correlation:.2f} correlation with financial performance.")
            
        return " ".join(summary_parts)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize analyzer
        analyzer = ResearchAnalyzer()
        
        # Example query
        query = "sustainability practices corporate financial performance"
        
        # Analyze papers
        results = await analyzer.analyze_sustainability_papers(query, max_papers=20)
        
        print("\n=== Analysis Results ===")
        print(f"Query: {results['metadata']['query']}")
        print(f"Papers analyzed: {results['metadata']['total_papers']}")
        print(f"Average sustainability score: {results['metadata']['avg_sustainability_score']:.2f}")
        print(f"Average profitability relevance: {results['metadata']['avg_profitability_relevance']:.2f}")
        
        print("\n=== Top Insights ===")
        for i, insight in enumerate(results['insights'][:3], 1):
            print(f"{i}. {insight.practice}")
            print(f"   Financial correlation: {insight.financial_correlation:.2f}")
            print(f"   Confidence: {insight.confidence_score:.2f}")
            print(f"   Description: {insight.impact_description}")
            print()
            
        print(f"\n=== Summary ===")
        print(results['summary'])
    
    # Run the example
    asyncio.run(main())