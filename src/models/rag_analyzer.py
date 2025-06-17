#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) System for SPR Analysis

This module implements a sophisticated RAG system that combines research paper
analysis with financial data to provide context-aware, research-backed insights.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
from collections import defaultdict
import sqlite3
import os

# Vector similarity imports (simplified implementation)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from research_processor.paper_analyzer import ResearchAnalyzer, SustainabilityInsight
from utils.text_processor import TextProcessor


@dataclass
class RetrievalResult:
    """Results from document retrieval"""
    document_id: str
    title: str
    content: str
    relevance_score: float
    retrieval_method: str
    metadata: Dict[str, Any]
    embedding_vector: List[float] = None


@dataclass
class GenerationResult:
    """Results from content generation"""
    generated_text: str
    confidence_score: float
    source_documents: List[RetrievalResult]
    generation_method: str
    context_relevance: float
    factual_accuracy: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGAnalysisResult:
    """Complete RAG analysis results"""
    query: str
    retrieval_results: List[RetrievalResult]
    generation_result: GenerationResult
    synthesis_quality: float
    research_coverage: float
    financial_alignment: float
    recommendations: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]


class ResearchKnowledgeBase:
    """
    Vector database for research papers and financial insights
    """
    
    def __init__(self, data_dir: str):
        """Initialize knowledge base with research documents"""
        self.data_dir = data_dir
        self.documents = {}
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=2
        )
        self.dimensionality_reducer = TruncatedSVD(n_components=100)
        self.document_index = {}
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Load and index all research documents"""
        try:
            await self._load_research_papers()
            await self._create_embeddings()
            await self._build_index()
            self.logger.info(f"Knowledge base initialized with {len(self.documents)} documents")
        except Exception as e:
            self.logger.error(f"Error initializing knowledge base: {str(e)}")
    
    async def _load_research_papers(self):
        """Load research papers from data directory"""
        research_dir = os.path.join(self.data_dir, 'raw', 'research_papers')
        
        if not os.path.exists(research_dir):
            self.logger.warning(f"Research directory not found: {research_dir}")
            return
        
        for filename in os.listdir(research_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(research_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract title and clean content
                    title = filename.replace('.txt', '').replace('_', ' ')
                    cleaned_content = self._clean_text(content)
                    
                    doc_id = f"research_{len(self.documents)}"
                    self.documents[doc_id] = {
                        'id': doc_id,
                        'title': title,
                        'content': cleaned_content,
                        'filename': filename,
                        'type': 'research_paper',
                        'word_count': len(cleaned_content.split()),
                        'created_date': datetime.now()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {str(e)}")
    
    async def _create_embeddings(self):
        """Create vector embeddings for all documents"""
        if not self.documents:
            return
        
        # Prepare document texts
        doc_texts = [doc['content'] for doc in self.documents.values()]
        doc_ids = list(self.documents.keys())
        
        # Create TF-IDF embeddings
        tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
        
        # Reduce dimensionality
        self.embeddings = self.dimensionality_reducer.fit_transform(tfidf_matrix)
        
        # Store embeddings in document metadata
        for i, doc_id in enumerate(doc_ids):
            self.documents[doc_id]['embedding'] = self.embeddings[i].tolist()
    
    async def _build_index(self):
        """Build searchable index for documents"""
        # Create keyword index
        self.keyword_index = defaultdict(set)
        
        for doc_id, doc in self.documents.items():
            # Index key terms from title and content
            text = (doc['title'] + ' ' + doc['content']).lower()
            words = re.findall(r'\b\w+\b', text)
            
            for word in words:
                if len(word) > 3:  # Skip short words
                    self.keyword_index[word].add(doc_id)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short lines (likely formatting artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 20]
        
        return '\n'.join(cleaned_lines)
    
    async def retrieve_documents(self, 
                               query: str, 
                               top_k: int = 5,
                               method: str = 'hybrid') -> List[RetrievalResult]:
        """Retrieve most relevant documents for a query"""
        
        if method == 'semantic':
            return await self._semantic_retrieval(query, top_k)
        elif method == 'keyword':
            return await self._keyword_retrieval(query, top_k)
        else:  # hybrid
            return await self._hybrid_retrieval(query, top_k)
    
    async def _semantic_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve documents using semantic similarity"""
        if not self.embeddings.any():
            return []
        
        # Vectorize query
        query_tfidf = self.vectorizer.transform([query])
        query_embedding = self.dimensionality_reducer.transform(query_tfidf)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        doc_ids = list(self.documents.keys())
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                doc_id = doc_ids[idx]
                doc = self.documents[doc_id]
                
                results.append(RetrievalResult(
                    document_id=doc_id,
                    title=doc['title'],
                    content=doc['content'][:1000],  # Truncate for efficiency
                    relevance_score=float(similarities[idx]),
                    retrieval_method='semantic_similarity',
                    metadata={
                        'word_count': doc['word_count'],
                        'filename': doc['filename'],
                        'similarity_score': float(similarities[idx])
                    },
                    embedding_vector=doc.get('embedding', [])
                ))
        
        return results
    
    async def _keyword_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve documents using keyword matching"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_words = {word for word in query_words if len(word) > 3}
        
        # Score documents by keyword overlap
        doc_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_id in self.keyword_index[word]:
                    doc_scores[doc_id] += 1.0 / len(query_words)
        
        # Sort by score and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            if score > 0:
                doc = self.documents[doc_id]
                
                results.append(RetrievalResult(
                    document_id=doc_id,
                    title=doc['title'],
                    content=doc['content'][:1000],
                    relevance_score=score,
                    retrieval_method='keyword_matching',
                    metadata={
                        'word_count': doc['word_count'],
                        'filename': doc['filename'],
                        'keyword_score': score
                    }
                ))
        
        return results
    
    async def _hybrid_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Combine semantic and keyword retrieval"""
        semantic_results = await self._semantic_retrieval(query, top_k)
        keyword_results = await self._keyword_retrieval(query, top_k)
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add semantic results with higher weight
        for result in semantic_results:
            combined_results[result.document_id] = result
            result.relevance_score *= 0.7  # Semantic weight
        
        # Add keyword results with lower weight
        for result in keyword_results:
            if result.document_id in combined_results:
                # Combine scores
                existing = combined_results[result.document_id]
                existing.relevance_score += result.relevance_score * 0.3
                existing.retrieval_method = 'hybrid'
            else:
                result.relevance_score *= 0.3  # Keyword weight
                result.retrieval_method = 'hybrid'
                combined_results[result.document_id] = result
        
        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x.relevance_score, 
            reverse=True
        )[:top_k]
        
        return sorted_results


class RAGSPRAnalyzer:
    """
    Main RAG system for SPR analysis combining retrieval and generation
    """
    
    def __init__(self, config: Dict[str, Any] = None, data_dir: str = None):
        """Initialize RAG analyzer"""
        self.config = config or {}
        self.data_dir = data_dir or 'data'
        self.knowledge_base = ResearchKnowledgeBase(self.data_dir)
        self.text_processor = TextProcessor()
        self.logger = self._setup_logging()
        
        # Generation parameters
        self.generation_params = {
            'max_context_length': 2000,
            'min_confidence_threshold': 0.3,
            'factual_accuracy_weight': 0.4,
            'context_relevance_weight': 0.6
        }
        
        # Research synthesis templates
        self.synthesis_templates = {
            'financial_analysis': "Based on research findings, the financial performance shows {findings}. Key insights from academic studies suggest {insights}.",
            'sustainability_impact': "Research indicates that sustainability practices {impact}. Academic literature supports {evidence}.",
            'risk_assessment': "Risk analysis from multiple studies reveals {risks}. Research consensus suggests {mitigation}.",
            'recommendations': "Based on comprehensive research analysis, we recommend {actions}. Supporting evidence includes {support}."
        }
    
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
    
    async def initialize(self):
        """Initialize the RAG system"""
        try:
            await self.knowledge_base.initialize()
            self.logger.info("RAG system initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing RAG system: {str(e)}")
    
    async def analyze_with_rag(self, 
                             query: str,
                             financial_data: Dict[str, Any] = None,
                             context: Dict[str, Any] = None) -> RAGAnalysisResult:
        """
        Main RAG analysis method combining retrieval and generation
        
        Args:
            query: Analysis question or topic
            financial_data: Company financial metrics
            context: Additional context (company, sector, etc.)
            
        Returns:
            Complete RAG analysis with research-backed insights
        """
        try:
            self.logger.info(f"Starting RAG analysis for query: {query[:100]}...")
            
            # Step 1: Retrieve relevant research documents
            retrieval_results = await self.knowledge_base.retrieve_documents(
                query, top_k=5, method='hybrid'
            )
            
            if not retrieval_results:
                self.logger.warning("No relevant documents found for query")
                return self._create_empty_result(query)
            
            # Step 2: Generate research-backed insights
            generation_result = await self._generate_insights(
                query, retrieval_results, financial_data, context
            )
            
            # Step 3: Calculate analysis quality metrics
            synthesis_quality = self._calculate_synthesis_quality(retrieval_results, generation_result)
            research_coverage = self._calculate_research_coverage(retrieval_results, query)
            financial_alignment = self._calculate_financial_alignment(generation_result, financial_data)
            
            # Step 4: Generate actionable recommendations
            recommendations = await self._generate_recommendations(
                generation_result, retrieval_results, financial_data
            )
            
            # Step 5: Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                retrieval_results, generation_result, financial_data
            )
            
            return RAGAnalysisResult(
                query=query,
                retrieval_results=retrieval_results,
                generation_result=generation_result,
                synthesis_quality=synthesis_quality,
                research_coverage=research_coverage,
                financial_alignment=financial_alignment,
                recommendations=recommendations,
                confidence_intervals=confidence_intervals
            )
            
        except Exception as e:
            self.logger.error(f"Error in RAG analysis: {str(e)}")
            return self._create_empty_result(query)
    
    async def _generate_insights(self, 
                               query: str,
                               retrieval_results: List[RetrievalResult],
                               financial_data: Dict[str, Any] = None,
                               context: Dict[str, Any] = None) -> GenerationResult:
        """Generate insights using retrieved research and financial data"""
        
        # Combine retrieved content for context
        research_context = self._combine_research_content(retrieval_results)
        
        # Generate insights based on query type
        if 'sustainability' in query.lower() or 'esg' in query.lower():
            generated_text = await self._generate_sustainability_insights(
                research_context, financial_data, context
            )
        elif 'financial' in query.lower() or 'profit' in query.lower():
            generated_text = await self._generate_financial_insights(
                research_context, financial_data, context
            )
        elif 'risk' in query.lower():
            generated_text = await self._generate_risk_insights(
                research_context, financial_data, context
            )
        else:
            generated_text = await self._generate_general_insights(
                research_context, financial_data, context
            )
        
        # Calculate generation quality metrics
        confidence_score = self._calculate_generation_confidence(retrieval_results, generated_text)
        context_relevance = self._calculate_context_relevance(query, generated_text)
        factual_accuracy = self._estimate_factual_accuracy(generated_text, research_context)
        
        return GenerationResult(
            generated_text=generated_text,
            confidence_score=confidence_score,
            source_documents=retrieval_results,
            generation_method='template_based_synthesis',
            context_relevance=context_relevance,
            factual_accuracy=factual_accuracy,
            metadata={
                'num_sources': len(retrieval_results),
                'context_length': len(research_context),
                'query_type': self._classify_query_type(query)
            }
        )
    
    def _combine_research_content(self, retrieval_results: List[RetrievalResult]) -> str:
        """Combine content from multiple research documents"""
        combined_content = []
        
        for result in retrieval_results:
            # Add document title and content snippet
            content_snippet = f"Research finding from '{result.title}': {result.content[:300]}..."
            combined_content.append(content_snippet)
        
        return '\n\n'.join(combined_content)
    
    async def _generate_sustainability_insights(self, 
                                              research_context: str,
                                              financial_data: Dict[str, Any],
                                              context: Dict[str, Any]) -> str:
        """Generate sustainability-focused insights"""
        
        # Extract key sustainability themes from research
        sustainability_themes = self._extract_sustainability_themes(research_context)
        
        # Analyze financial sustainability metrics
        financial_insights = self._analyze_sustainability_financials(financial_data) if financial_data else ""
        
        # Generate comprehensive analysis
        insights = []
        
        insights.append("## Sustainability Performance Analysis")
        insights.append(f"Research synthesis reveals key sustainability patterns: {', '.join(sustainability_themes[:3])}")
        
        if financial_insights:
            insights.append(f"Financial analysis shows: {financial_insights}")
        
        insights.append("## Research-Backed Findings")
        insights.append("Academic literature supports the following conclusions:")
        
        # Extract specific findings from research context
        findings = self._extract_key_findings(research_context, 'sustainability')
        for i, finding in enumerate(findings[:3], 1):
            insights.append(f"{i}. {finding}")
        
        return '\n\n'.join(insights)
    
    async def _generate_financial_insights(self, 
                                         research_context: str,
                                         financial_data: Dict[str, Any],
                                         context: Dict[str, Any]) -> str:
        """Generate financial performance insights"""
        
        insights = []
        insights.append("## Financial Performance Analysis")
        
        if financial_data:
            # Analyze key financial metrics
            financial_summary = self._summarize_financial_metrics(financial_data)
            insights.append(f"Current financial position: {financial_summary}")
        
        # Research-backed financial insights
        insights.append("## Research-Supported Analysis")
        financial_findings = self._extract_key_findings(research_context, 'financial')
        
        for i, finding in enumerate(financial_findings[:3], 1):
            insights.append(f"{i}. {finding}")
        
        return '\n\n'.join(insights)
    
    async def _generate_risk_insights(self, 
                                    research_context: str,
                                    financial_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> str:
        """Generate risk assessment insights"""
        
        insights = []
        insights.append("## Risk Assessment Analysis")
        
        # Identify risk factors from research
        risk_factors = self._extract_risk_factors(research_context)
        insights.append(f"Key risk factors identified: {', '.join(risk_factors[:3])}")
        
        if financial_data:
            financial_risks = self._assess_financial_risks(financial_data)
            insights.append(f"Financial risk assessment: {financial_risks}")
        
        return '\n\n'.join(insights)
    
    async def _generate_general_insights(self, 
                                       research_context: str,
                                       financial_data: Dict[str, Any],
                                       context: Dict[str, Any]) -> str:
        """Generate general comprehensive insights"""
        
        insights = []
        insights.append("## Comprehensive Analysis")
        insights.append("Research synthesis and financial analysis reveal:")
        
        # General findings from research
        general_findings = self._extract_key_findings(research_context, 'general')
        for i, finding in enumerate(general_findings[:3], 1):
            insights.append(f"{i}. {finding}")
        
        return '\n\n'.join(insights)
    
    def _extract_sustainability_themes(self, research_content: str) -> List[str]:
        """Extract sustainability themes from research content"""
        themes = []
        
        # Define sustainability keywords
        sustainability_keywords = [
            'environmental impact', 'carbon emissions', 'renewable energy',
            'social responsibility', 'governance practices', 'sustainable development',
            'green initiatives', 'circular economy', 'ESG performance'
        ]
        
        content_lower = research_content.lower()
        for keyword in sustainability_keywords:
            if keyword in content_lower:
                themes.append(keyword)
        
        return themes
    
    def _extract_key_findings(self, research_content: str, focus: str = 'general') -> List[str]:
        """Extract key findings from research content"""
        findings = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', research_content)
        
        # Keywords for different focus areas
        focus_keywords = {
            'sustainability': ['sustainable', 'environmental', 'social', 'governance', 'ESG'],
            'financial': ['profit', 'revenue', 'financial', 'economic', 'cost'],
            'general': ['research', 'study', 'analysis', 'findings', 'results']
        }
        
        keywords = focus_keywords.get(focus, focus_keywords['general'])
        
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 30 and any(keyword in sentence.lower() for keyword in keywords):
                findings.append(sentence)
                if len(findings) >= 5:  # Limit findings
                    break
        
        return findings
    
    def _analyze_sustainability_financials(self, financial_data: Dict[str, Any]) -> str:
        """Analyze financial data from sustainability perspective"""
        insights = []
        
        if 'sustainability_metrics' in financial_data:
            sustainability = financial_data['sustainability_metrics']
            
            if isinstance(sustainability, dict):
                esg_score = sustainability.get('esg_score', 0)
                if esg_score > 70:
                    insights.append("strong ESG performance")
                elif esg_score > 50:
                    insights.append("moderate ESG performance")
                else:
                    insights.append("opportunities for ESG improvement")
        
        return ', '.join(insights) if insights else "sustainability metrics require further analysis"
    
    def _summarize_financial_metrics(self, financial_data: Dict[str, Any]) -> str:
        """Summarize key financial metrics"""
        summary = []
        
        if 'financial_metrics' in financial_data:
            financials = financial_data['financial_metrics']
            
            if isinstance(financials, dict):
                # Check profitability
                profit_margin = financials.get('profit_margin', 0)
                if profit_margin > 0.15:
                    summary.append("strong profitability")
                elif profit_margin > 0.05:
                    summary.append("moderate profitability")
                else:
                    summary.append("profitability concerns")
                
                # Check growth
                revenue_growth = financials.get('revenue_growth', 0)
                if revenue_growth > 0.1:
                    summary.append("strong growth")
                elif revenue_growth > 0:
                    summary.append("positive growth")
                else:
                    summary.append("declining revenue")
        
        return ', '.join(summary) if summary else "mixed financial performance"
    
    def _extract_risk_factors(self, research_content: str) -> List[str]:
        """Extract risk factors from research content"""
        risk_keywords = [
            'risk', 'volatility', 'uncertainty', 'challenge', 'threat',
            'regulatory', 'compliance', 'market risk', 'operational risk'
        ]
        
        risk_factors = []
        content_lower = research_content.lower()
        
        for keyword in risk_keywords:
            if keyword in content_lower:
                risk_factors.append(keyword)
        
        return risk_factors
    
    def _assess_financial_risks(self, financial_data: Dict[str, Any]) -> str:
        """Assess financial risks from data"""
        risks = []
        
        if 'financial_metrics' in financial_data:
            financials = financial_data['financial_metrics']
            
            if isinstance(financials, dict):
                debt_ratio = financials.get('debt_to_equity', 0)
                if debt_ratio > 1.0:
                    risks.append("high debt levels")
                
                current_ratio = financials.get('current_ratio', 1)
                if current_ratio < 1.0:
                    risks.append("liquidity concerns")
        
        return ', '.join(risks) if risks else "manageable financial risk profile"
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for appropriate response generation"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sustainability', 'esg', 'environmental', 'social']):
            return 'sustainability'
        elif any(word in query_lower for word in ['financial', 'profit', 'revenue', 'performance']):
            return 'financial'
        elif any(word in query_lower for word in ['risk', 'threat', 'challenge']):
            return 'risk'
        else:
            return 'general'
    
    def _calculate_generation_confidence(self, retrieval_results: List[RetrievalResult], generated_text: str) -> float:
        """Calculate confidence in generated insights"""
        if not retrieval_results:
            return 0.0
        
        # Base confidence on retrieval quality and text length
        avg_relevance = np.mean([r.relevance_score for r in retrieval_results])
        text_quality = min(len(generated_text) / 500, 1.0)  # Normalize by expected length
        
        confidence = (avg_relevance * 0.6) + (text_quality * 0.4)
        return min(confidence, 1.0)
    
    def _calculate_context_relevance(self, query: str, generated_text: str) -> float:
        """Calculate how relevant the generated text is to the query"""
        query_words = set(query.lower().split())
        text_words = set(generated_text.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(text_words))
        relevance = overlap / len(query_words) if query_words else 0.0
        
        return min(relevance, 1.0)
    
    def _estimate_factual_accuracy(self, generated_text: str, research_context: str) -> float:
        """Estimate factual accuracy based on research support"""
        # Simple heuristic: check for research-backed statements
        research_indicators = ['research shows', 'studies indicate', 'evidence suggests', 'findings reveal']
        
        accuracy_score = 0.5  # Base score
        
        for indicator in research_indicators:
            if indicator in generated_text.lower():
                accuracy_score += 0.1
        
        return min(accuracy_score, 1.0)
    
    def _calculate_synthesis_quality(self, retrieval_results: List[RetrievalResult], generation_result: GenerationResult) -> float:
        """Calculate overall synthesis quality"""
        if not retrieval_results:
            return 0.0
        
        # Combine multiple quality metrics
        retrieval_quality = np.mean([r.relevance_score for r in retrieval_results])
        generation_quality = generation_result.confidence_score
        factual_quality = generation_result.factual_accuracy
        
        synthesis_quality = (retrieval_quality * 0.4) + (generation_quality * 0.3) + (factual_quality * 0.3)
        return min(synthesis_quality, 1.0)
    
    def _calculate_research_coverage(self, retrieval_results: List[RetrievalResult], query: str) -> float:
        """Calculate how well the research covers the query topic"""
        if not retrieval_results:
            return 0.0
        
        # Check diversity of sources and relevance scores
        avg_relevance = np.mean([r.relevance_score for r in retrieval_results])
        source_diversity = len(set(r.document_id for r in retrieval_results)) / len(retrieval_results)
        
        coverage = (avg_relevance * 0.7) + (source_diversity * 0.3)
        return min(coverage, 1.0)
    
    def _calculate_financial_alignment(self, generation_result: GenerationResult, financial_data: Dict[str, Any]) -> float:
        """Calculate alignment between generated insights and financial data"""
        if not financial_data:
            return 0.5  # Neutral if no financial data
        
        # Check if generated text mentions financial metrics
        text_lower = generation_result.generated_text.lower()
        financial_terms = ['profit', 'revenue', 'financial', 'performance', 'growth', 'margin']
        
        term_coverage = sum(1 for term in financial_terms if term in text_lower) / len(financial_terms)
        
        return min(term_coverage + 0.3, 1.0)  # Add base alignment
    
    async def _generate_recommendations(self, 
                                      generation_result: GenerationResult,
                                      retrieval_results: List[RetrievalResult],
                                      financial_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Extract actionable insights from generated text
        text = generation_result.generated_text.lower()
        
        if 'sustainability' in text:
            recommendations.append("Enhance sustainability reporting and ESG metrics tracking")
        
        if 'risk' in text:
            recommendations.append("Implement comprehensive risk management strategies")
        
        if 'financial' in text and financial_data:
            recommendations.append("Optimize financial performance through data-driven strategies")
        
        if len(retrieval_results) > 3:
            recommendations.append("Leverage academic research insights for strategic planning")
        
        return recommendations or ["Continue monitoring performance and research developments"]
    
    def _calculate_confidence_intervals(self, 
                                      retrieval_results: List[RetrievalResult],
                                      generation_result: GenerationResult,
                                      financial_data: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        base_confidence = generation_result.confidence_score
        
        # Simple confidence intervals (in practice, would use more sophisticated methods)
        intervals = {
            'synthesis_quality': (max(base_confidence - 0.15, 0), min(base_confidence + 0.15, 1)),
            'research_relevance': (max(base_confidence - 0.1, 0), min(base_confidence + 0.1, 1)),
            'recommendation_reliability': (max(base_confidence - 0.2, 0), min(base_confidence + 0.2, 1))
        }
        
        return intervals
    
    def _create_empty_result(self, query: str) -> RAGAnalysisResult:
        """Create empty result when analysis fails"""
        return RAGAnalysisResult(
            query=query,
            retrieval_results=[],
            generation_result=GenerationResult(
                generated_text="No relevant research found for this query.",
                confidence_score=0.0,
                source_documents=[],
                generation_method="fallback",
                context_relevance=0.0,
                factual_accuracy=0.0
            ),
            synthesis_quality=0.0,
            research_coverage=0.0,
            financial_alignment=0.0,
            recommendations=["Consider expanding the query or checking data availability"],
            confidence_intervals={}
        )

    def get_pipeline_stage_quality(self, rag_result: RAGAnalysisResult) -> float:
        """Calculate quality score for the RAG pipeline stage"""
        if not rag_result or not rag_result.retrieval_results:
            return 0.0
        
        # Combine multiple quality indicators
        retrieval_quality = np.mean([r.relevance_score for r in rag_result.retrieval_results])
        synthesis_quality = rag_result.synthesis_quality
        research_coverage = rag_result.research_coverage
        
        overall_quality = (retrieval_quality * 0.4) + (synthesis_quality * 0.3) + (research_coverage * 0.3)
        return min(overall_quality, 1.0)
