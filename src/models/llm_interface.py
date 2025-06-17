#!/usr/bin/env python3
"""
Large Language Model Interface for SPR Analysis

This module provides a natural language interface for SPR analysis,
enabling conversational queries and automated diagnostic reporting.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from models.rag_analyzer import RAGSPRAnalyzer, RAGAnalysisResult


class QueryType(Enum):
    FINANCIAL_ANALYSIS = "financial_analysis"
    SUSTAINABILITY_ASSESSMENT = "sustainability_assessment"
    RISK_EVALUATION = "risk_evaluation"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    DIAGNOSTIC_QUERY = "diagnostic_query"
    GENERAL_INQUIRY = "general_inquiry"


class ResponseStyle(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"
    EXECUTIVE = "executive"
    DETAILED = "detailed"
    SUMMARY = "summary"


@dataclass
class NLQuery:
    """Natural language query structure"""
    raw_query: str
    processed_query: str
    query_type: QueryType
    entities: List[str]
    intent: str
    context: Dict[str, Any]
    response_style: ResponseStyle
    confidence: float


@dataclass
class LLMResponse:
    """LLM response structure"""
    query: NLQuery
    response_text: str
    supporting_data: Dict[str, Any]
    confidence_score: float
    source_quality: float
    response_type: str
    recommendations: List[str]
    follow_up_questions: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NLPProcessor:
    """
    Natural Language Processing for query understanding and response generation
    """
    
    def __init__(self):
        """Initialize NLP processor"""
        self.financial_keywords = [
            'profit', 'revenue', 'margin', 'growth', 'roi', 'performance',
            'earnings', 'cash flow', 'debt', 'equity', 'assets', 'liabilities'
        ]
        
        self.sustainability_keywords = [
            'sustainability', 'esg', 'environmental', 'social', 'governance',
            'carbon', 'emissions', 'renewable', 'green', 'responsible', 'ethical'
        ]
        
        self.risk_keywords = [
            'risk', 'threat', 'vulnerability', 'exposure', 'volatility',
            'uncertainty', 'compliance', 'regulatory', 'operational'
        ]
        
        self.comparative_keywords = [
            'compare', 'versus', 'against', 'benchmark', 'relative',
            'better', 'worse', 'higher', 'lower', 'similar'
        ]
        
        self.diagnostic_keywords = [
            'why', 'how', 'what', 'explain', 'analyze', 'diagnose',
            'cause', 'reason', 'factor', 'issue', 'problem'
        ]
        
        # Response style indicators
        self.style_indicators = {
            ResponseStyle.TECHNICAL: ['technical', 'detailed', 'analysis', 'metrics'],
            ResponseStyle.BUSINESS: ['business', 'commercial', 'strategy', 'market'],
            ResponseStyle.EXECUTIVE: ['executive', 'summary', 'overview', 'brief'],
            ResponseStyle.DETAILED: ['detailed', 'comprehensive', 'thorough', 'complete'],
            ResponseStyle.SUMMARY: ['summary', 'brief', 'quick', 'overview']
        }
    
    def parse_query(self, query: str, context: Dict[str, Any] = None) -> NLQuery:
        """Parse natural language query into structured format"""
        
        # Clean and normalize query
        processed_query = self._preprocess_query(query)
        
        # Extract entities (companies, metrics, etc.)
        entities = self._extract_entities(processed_query)
        
        # Classify query type
        query_type = self._classify_query_type(processed_query)
        
        # Extract intent
        intent = self._extract_intent(processed_query, query_type)
        
        # Determine response style
        response_style = self._determine_response_style(processed_query)
        
        # Calculate parsing confidence
        confidence = self._calculate_parsing_confidence(processed_query, entities, query_type)
        
        return NLQuery(
            raw_query=query,
            processed_query=processed_query,
            query_type=query_type,
            entities=entities,
            intent=intent,
            context=context or {},
            response_style=response_style,
            confidence=confidence
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the query text"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove common question words for processing
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        # Keep question words but normalize
        
        return query
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities like company names, metrics, etc."""
        entities = []
        
        # Common stock symbols pattern
        stock_pattern = r'\b[A-Z]{1,5}\b'
        stock_matches = re.findall(stock_pattern, query.upper())
        entities.extend(stock_matches)
        
        # Financial metrics
        metric_patterns = [
            r'profit\s+margin', r'debt\s+to\s+equity', r'return\s+on\s+equity',
            r'revenue\s+growth', r'current\s+ratio', r'esg\s+score'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Company names (simplified - would use NER in production)
        common_companies = ['apple', 'microsoft', 'tesla', 'amazon', 'google']
        for company in common_companies:
            if company in query:
                entities.append(company)
        
        return list(set(entities))
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        
        # Count keyword matches for each category
        financial_score = sum(1 for keyword in self.financial_keywords if keyword in query)
        sustainability_score = sum(1 for keyword in self.sustainability_keywords if keyword in query)
        risk_score = sum(1 for keyword in self.risk_keywords if keyword in query)
        comparative_score = sum(1 for keyword in self.comparative_keywords if keyword in query)
        diagnostic_score = sum(1 for keyword in self.diagnostic_keywords if keyword in query)
        
        # Determine primary category
        scores = {
            QueryType.FINANCIAL_ANALYSIS: financial_score,
            QueryType.SUSTAINABILITY_ASSESSMENT: sustainability_score,
            QueryType.RISK_EVALUATION: risk_score,
            QueryType.COMPARATIVE_ANALYSIS: comparative_score,
            QueryType.DIAGNOSTIC_QUERY: diagnostic_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.GENERAL_INQUIRY
        
        # Return the category with highest score
        for query_type, score in scores.items():
            if score == max_score:
                return query_type
        
        return QueryType.GENERAL_INQUIRY
    
    def _extract_intent(self, query: str, query_type: QueryType) -> str:
        """Extract the specific intent of the query"""
        
        # Intent patterns based on query type
        intent_patterns = {
            QueryType.FINANCIAL_ANALYSIS: {
                'performance_evaluation': ['performance', 'evaluate', 'assess'],
                'metric_calculation': ['calculate', 'compute', 'determine'],
                'trend_analysis': ['trend', 'pattern', 'change', 'over time']
            },
            QueryType.SUSTAINABILITY_ASSESSMENT: {
                'esg_evaluation': ['esg', 'sustainability score', 'rating'],
                'impact_assessment': ['impact', 'effect', 'consequence'],
                'compliance_check': ['compliance', 'standards', 'guidelines']
            },
            QueryType.RISK_EVALUATION: {
                'risk_identification': ['identify', 'find', 'detect'],
                'risk_quantification': ['quantify', 'measure', 'calculate'],
                'mitigation_strategies': ['mitigate', 'reduce', 'manage']
            }
        }
        
        if query_type in intent_patterns:
            for intent, keywords in intent_patterns[query_type].items():
                if any(keyword in query for keyword in keywords):
                    return intent
        
        return 'general_analysis'
    
    def _determine_response_style(self, query: str) -> ResponseStyle:
        """Determine appropriate response style"""
        
        style_scores = {}
        for style, indicators in self.style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query)
            style_scores[style] = score
        
        max_score = max(style_scores.values())
        if max_score == 0:
            return ResponseStyle.BUSINESS  # Default style
        
        for style, score in style_scores.items():
            if score == max_score:
                return style
        
        return ResponseStyle.BUSINESS
    
    def _calculate_parsing_confidence(self, query: str, entities: List[str], query_type: QueryType) -> float:
        """Calculate confidence in query parsing"""
        
        # Base confidence on query clarity and entity extraction
        entity_score = min(len(entities) / 3, 1.0)  # Normalize by expected entities
        
        # Query clarity based on length and structure
        clarity_score = min(len(query.split()) / 10, 1.0)  # Normalize by expected length
        
        # Type classification confidence
        type_confidence = 0.8 if query_type != QueryType.GENERAL_INQUIRY else 0.4
        
        overall_confidence = (entity_score * 0.3) + (clarity_score * 0.3) + (type_confidence * 0.4)
        return min(overall_confidence, 1.0)


class LLMInterface:
    """
    Main LLM interface for natural language SPR analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None, data_dir: str = None):
        """Initialize LLM interface"""
        self.config = config or {}
        self.data_dir = data_dir or 'data'
        self.nlp_processor = NLPProcessor()
        self.rag_analyzer = RAGSPRAnalyzer(config, data_dir)
        self.logger = self._setup_logging()
        
        # Response generation templates
        self.response_templates = {
            QueryType.FINANCIAL_ANALYSIS: {
                ResponseStyle.TECHNICAL: "Technical financial analysis indicates {analysis}. Key metrics show {metrics}.",
                ResponseStyle.BUSINESS: "From a business perspective, {analysis}. This suggests {implications}.",
                ResponseStyle.EXECUTIVE: "Executive Summary: {summary}. Recommendation: {action}.",
                ResponseStyle.DETAILED: "Detailed Analysis:\n\n{detailed_analysis}\n\nSupporting Data:\n{data}",
                ResponseStyle.SUMMARY: "Summary: {brief_analysis}. Key takeaway: {conclusion}."
            },
            QueryType.SUSTAINABILITY_ASSESSMENT: {
                ResponseStyle.TECHNICAL: "ESG technical assessment reveals {esg_analysis}. Metrics: {esg_metrics}.",
                ResponseStyle.BUSINESS: "Sustainability performance shows {sustainability_summary}. Business impact: {impact}.",
                ResponseStyle.EXECUTIVE: "ESG Overview: {esg_summary}. Strategic recommendation: {strategy}.",
                ResponseStyle.DETAILED: "Comprehensive ESG Analysis:\n\n{detailed_esg}\n\nBenchmarking:\n{benchmarks}",
                ResponseStyle.SUMMARY: "ESG Summary: {esg_brief}. Overall rating: {rating}."
            }
        }
        
        # Conversation context
        self.conversation_history = []
        self.context_memory = {}
    
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
        """Initialize the LLM interface and dependencies"""
        try:
            await self.rag_analyzer.initialize()
            self.logger.info("LLM interface initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing LLM interface: {str(e)}")
    
    async def process_query(self, 
                          query: str,
                          company_data: Dict[str, Any] = None,
                          context: Dict[str, Any] = None) -> LLMResponse:
        """
        Process a natural language query and generate a response
        
        Args:
            query: Natural language query from user
            company_data: Company financial/sustainability data
            context: Additional context (company symbol, sector, etc.)
            
        Returns:
            Structured LLM response with analysis and recommendations
        """
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Parse and understand the query
            parsed_query = self.nlp_processor.parse_query(query, context)
            
            # Step 2: Retrieve relevant information using RAG
            rag_result = await self._retrieve_relevant_info(parsed_query, company_data)
            
            # Step 3: Generate response based on query type and style
            response_text = await self._generate_response(parsed_query, rag_result, company_data)
            
            # Step 4: Calculate response quality metrics
            confidence_score = self._calculate_response_confidence(parsed_query, rag_result)
            source_quality = self._assess_source_quality(rag_result)
            
            # Step 5: Generate recommendations and follow-up questions
            recommendations = await self._generate_recommendations(parsed_query, rag_result, company_data)
            follow_up_questions = self._generate_follow_up_questions(parsed_query, rag_result)
            
            # Step 6: Update conversation context
            self._update_conversation_context(parsed_query, response_text)
            
            return LLMResponse(
                query=parsed_query,
                response_text=response_text,
                supporting_data=self._extract_supporting_data(rag_result, company_data),
                confidence_score=confidence_score,
                source_quality=source_quality,
                response_type=parsed_query.query_type.value,
                recommendations=recommendations,
                follow_up_questions=follow_up_questions,
                metadata={
                    'processing_time': datetime.now().isoformat(),
                    'rag_sources': len(rag_result.retrieval_results) if rag_result else 0,
                    'conversation_turn': len(self.conversation_history)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self._create_error_response(query, str(e))
    
    async def _retrieve_relevant_info(self, 
                                    parsed_query: NLQuery, 
                                    company_data: Dict[str, Any]) -> Optional[RAGAnalysisResult]:
        """Retrieve relevant information using RAG system"""
        
        try:
            # Construct enhanced query for RAG system
            enhanced_query = self._enhance_query_for_rag(parsed_query)
            
            # Perform RAG analysis
            rag_result = await self.rag_analyzer.analyze_with_rag(
                query=enhanced_query,
                financial_data=company_data,
                context=parsed_query.context
            )
            
            return rag_result
            
        except Exception as e:
            self.logger.error(f"Error in information retrieval: {str(e)}")
            return None
    
    def _enhance_query_for_rag(self, parsed_query: NLQuery) -> str:
        """Enhance query with context and entities for better RAG retrieval"""
        
        enhanced_parts = [parsed_query.processed_query]
        
        # Add entities to improve retrieval
        if parsed_query.entities:
            enhanced_parts.append(f"Related to: {', '.join(parsed_query.entities)}")
        
        # Add query type context
        type_context = {
            QueryType.FINANCIAL_ANALYSIS: "financial performance profitability revenue",
            QueryType.SUSTAINABILITY_ASSESSMENT: "sustainability ESG environmental social governance",
            QueryType.RISK_EVALUATION: "risk assessment management mitigation",
            QueryType.COMPARATIVE_ANALYSIS: "comparison benchmark industry peer analysis",
            QueryType.DIAGNOSTIC_QUERY: "analysis explanation causation factors"
        }
        
        if parsed_query.query_type in type_context:
            enhanced_parts.append(type_context[parsed_query.query_type])
        
        return ' '.join(enhanced_parts)
    
    async def _generate_response(self, 
                               parsed_query: NLQuery,
                               rag_result: Optional[RAGAnalysisResult],
                               company_data: Dict[str, Any]) -> str:
        """Generate natural language response based on query and retrieved information"""
        
        # Select appropriate response template
        template = self._select_response_template(parsed_query)
        
        if not template:
            return await self._generate_fallback_response(parsed_query, rag_result, company_data)
        
        # Extract information for template filling
        template_data = await self._extract_template_data(parsed_query, rag_result, company_data)
        
        # Fill template with data
        try:
            response = template.format(**template_data)
            return response
        except KeyError as e:
            self.logger.warning(f"Template formatting error: {str(e)}")
            return await self._generate_fallback_response(parsed_query, rag_result, company_data)
    
    def _select_response_template(self, parsed_query: NLQuery) -> Optional[str]:
        """Select appropriate response template based on query type and style"""
        
        if parsed_query.query_type in self.response_templates:
            style_templates = self.response_templates[parsed_query.query_type]
            if parsed_query.response_style in style_templates:
                return style_templates[parsed_query.response_style]
        
        return None
    
    async def _extract_template_data(self, 
                                   parsed_query: NLQuery,
                                   rag_result: Optional[RAGAnalysisResult],
                                   company_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract data to fill response templates"""
        
        template_data = {}
        
        # Extract analysis from RAG result
        if rag_result and rag_result.generation_result:
            analysis_text = rag_result.generation_result.generated_text
            
            # Parse analysis into template fields
            template_data['analysis'] = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
            template_data['summary'] = analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text
            template_data['brief_analysis'] = analysis_text[:150] + "..." if len(analysis_text) > 150 else analysis_text
            template_data['detailed_analysis'] = analysis_text
        else:
            template_data.update({
                'analysis': 'Analysis data not available',
                'summary': 'Summary not available',
                'brief_analysis': 'Brief analysis not available',
                'detailed_analysis': 'Detailed analysis not available'
            })
        
        # Extract financial metrics
        if company_data and 'financial_metrics' in company_data:
            financial_metrics = company_data['financial_metrics']
            metrics_summary = self._summarize_financial_metrics(financial_metrics)
            template_data['metrics'] = metrics_summary
        else:
            template_data['metrics'] = 'Financial metrics not available'
        
        # Extract ESG/sustainability data
        if company_data and 'sustainability_metrics' in company_data:
            sustainability_metrics = company_data['sustainability_metrics']
            esg_summary = self._summarize_sustainability_metrics(sustainability_metrics)
            template_data.update({
                'esg_analysis': esg_summary,
                'esg_metrics': esg_summary,
                'sustainability_summary': esg_summary,
                'esg_summary': esg_summary,
                'esg_brief': esg_summary[:100] + "..." if len(esg_summary) > 100 else esg_summary,
                'detailed_esg': esg_summary
            })
        else:
            esg_default = 'ESG metrics not available'
            template_data.update({
                'esg_analysis': esg_default,
                'esg_metrics': esg_default,
                'sustainability_summary': esg_default,
                'esg_summary': esg_default,
                'esg_brief': esg_default,
                'detailed_esg': esg_default
            })
        
        # Add default values for common template fields
        template_data.update({
            'implications': 'Further analysis recommended',
            'action': 'Continue monitoring and analysis',
            'strategy': 'Implement best practices',
            'impact': 'Impact assessment pending',
            'data': 'Supporting data available upon request',
            'benchmarks': 'Industry benchmarking in progress',
            'rating': 'Rating calculation in progress',
            'conclusion': 'Analysis complete'
        })
        
        return template_data
    
    async def _generate_fallback_response(self, 
                                        parsed_query: NLQuery,
                                        rag_result: Optional[RAGAnalysisResult],
                                        company_data: Dict[str, Any]) -> str:
        """Generate fallback response when templates fail"""
        
        response_parts = []
        
        # Basic response based on query type
        if parsed_query.query_type == QueryType.FINANCIAL_ANALYSIS:
            response_parts.append("Financial analysis has been performed.")
            if company_data and 'financial_metrics' in company_data:
                response_parts.append(f"Key insights: {self._summarize_financial_metrics(company_data['financial_metrics'])}")
        
        elif parsed_query.query_type == QueryType.SUSTAINABILITY_ASSESSMENT:
            response_parts.append("Sustainability assessment has been completed.")
            if company_data and 'sustainability_metrics' in company_data:
                response_parts.append(f"ESG insights: {self._summarize_sustainability_metrics(company_data['sustainability_metrics'])}")
        
        else:
            response_parts.append("Analysis has been performed based on available data.")
        
        # Add RAG insights if available
        if rag_result and rag_result.generation_result:
            response_parts.append(f"Research insights: {rag_result.generation_result.generated_text[:200]}...")
        
        return '\n\n'.join(response_parts)
    
    def _summarize_financial_metrics(self, financial_metrics: Dict[str, Any]) -> str:
        """Summarize financial metrics for response"""
        if not isinstance(financial_metrics, dict):
            return "Financial metrics data format issue"
        
        summary_parts = []
        
        # Key financial indicators
        profit_margin = financial_metrics.get('profit_margin', 0)
        if profit_margin > 0.15:
            summary_parts.append("strong profitability")
        elif profit_margin > 0.05:
            summary_parts.append("moderate profitability")
        else:
            summary_parts.append("profitability concerns")
        
        revenue_growth = financial_metrics.get('revenue_growth', 0)
        if revenue_growth > 0.1:
            summary_parts.append("strong growth")
        elif revenue_growth > 0:
            summary_parts.append("positive growth")
        else:
            summary_parts.append("declining revenue")
        
        return ', '.join(summary_parts) if summary_parts else "mixed financial performance"
    
    def _summarize_sustainability_metrics(self, sustainability_metrics: Dict[str, Any]) -> str:
        """Summarize sustainability metrics for response"""
        if not isinstance(sustainability_metrics, dict):
            return "Sustainability metrics data format issue"
        
        summary_parts = []
        
        esg_score = sustainability_metrics.get('esg_score', 0)
        if esg_score > 70:
            summary_parts.append("strong ESG performance")
        elif esg_score > 50:
            summary_parts.append("moderate ESG performance")
        else:
            summary_parts.append("ESG improvement opportunities")
        
        env_score = sustainability_metrics.get('environmental_score', 0)
        if env_score > 70:
            summary_parts.append("good environmental practices")
        
        return ', '.join(summary_parts) if summary_parts else "ESG assessment in progress"
    
    def _calculate_response_confidence(self, 
                                     parsed_query: NLQuery, 
                                     rag_result: Optional[RAGAnalysisResult]) -> float:
        """Calculate confidence in the generated response"""
        
        query_confidence = parsed_query.confidence
        
        rag_confidence = 0.5  # Default
        if rag_result and rag_result.generation_result:
            rag_confidence = rag_result.generation_result.confidence_score
        
        # Combine confidences
        overall_confidence = (query_confidence * 0.4) + (rag_confidence * 0.6)
        return min(overall_confidence, 1.0)
    
    def _assess_source_quality(self, rag_result: Optional[RAGAnalysisResult]) -> float:
        """Assess quality of information sources"""
        if not rag_result or not rag_result.retrieval_results:
            return 0.0
        
        # Average relevance of retrieved sources
        avg_relevance = np.mean([r.relevance_score for r in rag_result.retrieval_results])
        return min(avg_relevance, 1.0)
    
    async def _generate_recommendations(self, 
                                      parsed_query: NLQuery,
                                      rag_result: Optional[RAGAnalysisResult],
                                      company_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Query-specific recommendations
        if parsed_query.query_type == QueryType.FINANCIAL_ANALYSIS:
            recommendations.append("Monitor key financial metrics regularly")
            recommendations.append("Consider benchmarking against industry peers")
        
        elif parsed_query.query_type == QueryType.SUSTAINABILITY_ASSESSMENT:
            recommendations.append("Enhance ESG reporting and disclosure")
            recommendations.append("Implement sustainability best practices")
        
        elif parsed_query.query_type == QueryType.RISK_EVALUATION:
            recommendations.append("Develop comprehensive risk management framework")
            recommendations.append("Regular risk assessment and monitoring")
        
        # RAG-based recommendations
        if rag_result and rag_result.recommendations:
            recommendations.extend(rag_result.recommendations[:2])  # Limit to top 2
        
        return recommendations or ["Continue regular analysis and monitoring"]
    
    def _generate_follow_up_questions(self, 
                                    parsed_query: NLQuery, 
                                    rag_result: Optional[RAGAnalysisResult]) -> List[str]:
        """Generate follow-up questions for deeper analysis"""
        
        follow_ups = []
        
        # Query-type specific follow-ups
        if parsed_query.query_type == QueryType.FINANCIAL_ANALYSIS:
            follow_ups.extend([
                "How does this compare to industry benchmarks?",
                "What are the key drivers of financial performance?",
                "What trends do you see over the past 5 years?"
            ])
        
        elif parsed_query.query_type == QueryType.SUSTAINABILITY_ASSESSMENT:
            follow_ups.extend([
                "How can ESG performance be improved?",
                "What are the material sustainability issues?",
                "How does this compare to sustainability leaders?"
            ])
        
        elif parsed_query.query_type == QueryType.RISK_EVALUATION:
            follow_ups.extend([
                "What mitigation strategies would you recommend?",
                "How has risk profile changed over time?",
                "What are the emerging risk factors?"
            ])
        
        # Limit to top 3 follow-ups
        return follow_ups[:3]
    
    def _extract_supporting_data(self, 
                               rag_result: Optional[RAGAnalysisResult], 
                               company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract supporting data for the response"""
        
        supporting_data = {}
        
        # RAG analysis data
        if rag_result:
            supporting_data['rag_analysis'] = {
                'synthesis_quality': rag_result.synthesis_quality,
                'research_coverage': rag_result.research_coverage,
                'financial_alignment': rag_result.financial_alignment,
                'confidence_intervals': rag_result.confidence_intervals
            }
            
            # Source information
            supporting_data['sources'] = [
                {
                    'title': r.title,
                    'relevance': r.relevance_score,
                    'method': r.retrieval_method
                }
                for r in rag_result.retrieval_results[:3]  # Top 3 sources
            ]
        
        # Company data summary
        if company_data:
            supporting_data['company_data'] = {
                'financial_available': 'financial_metrics' in company_data,
                'sustainability_available': 'sustainability_metrics' in company_data,
                'data_quality': 'good' if company_data else 'limited'
            }
        
        return supporting_data
    
    def _update_conversation_context(self, parsed_query: NLQuery, response_text: str):
        """Update conversation context for future queries"""
        
        # Add to conversation history
        self.conversation_history.append({
            'query': parsed_query.raw_query,
            'query_type': parsed_query.query_type.value,
            'response': response_text[:200],  # Truncated for memory
            'timestamp': datetime.now().isoformat(),
            'entities': parsed_query.entities
        })
        
        # Update context memory
        for entity in parsed_query.entities:
            if entity not in self.context_memory:
                self.context_memory[entity] = []
            
            self.context_memory[entity].append({
                'query_type': parsed_query.query_type.value,
                'timestamp': datetime.now().isoformat()
            })
        
        # Limit conversation history to last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _create_error_response(self, query: str, error_message: str) -> LLMResponse:
        """Create error response when processing fails"""
        
        error_query = NLQuery(
            raw_query=query,
            processed_query=query,
            query_type=QueryType.GENERAL_INQUIRY,
            entities=[],
            intent='error_handling',
            context={},
            response_style=ResponseStyle.BUSINESS,
            confidence=0.0
        )
        
        return LLMResponse(
            query=error_query,
            response_text=f"I apologize, but I encountered an error processing your query: {error_message}. Please try rephrasing your question or check the data availability.",
            supporting_data={},
            confidence_score=0.0,
            source_quality=0.0,
            response_type='error',
            recommendations=["Try rephrasing the query", "Check data availability", "Contact support if issue persists"],
            follow_up_questions=["Would you like to try a different question?"],
            metadata={'error': error_message}
        )
    
    async def generate_diagnostic_report(self, 
                                       company_symbol: str,
                                       company_data: Dict[str, Any] = None) -> str:
        """Generate comprehensive diagnostic report"""
        
        try:
            self.logger.info(f"Generating diagnostic report for {company_symbol}")
            
            # Predefined diagnostic queries
            diagnostic_queries = [
                f"Analyze the financial performance of {company_symbol}",
                f"Assess the sustainability and ESG performance of {company_symbol}",
                f"Evaluate the risk factors for {company_symbol}",
                f"What are the key strengths and weaknesses of {company_symbol}?"
            ]
            
            report_sections = []
            report_sections.append(f"# Comprehensive Diagnostic Report: {company_symbol}")
            report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_sections.append("")
            
            # Process each diagnostic query
            for i, query in enumerate(diagnostic_queries, 1):
                response = await self.process_query(query, company_data, {'symbol': company_symbol})
                
                section_title = f"## Section {i}: {query}"
                report_sections.append(section_title)
                report_sections.append(response.response_text)
                
                if response.recommendations:
                    report_sections.append("### Recommendations:")
                    for rec in response.recommendations:
                        report_sections.append(f"- {rec}")
                
                report_sections.append("")
            
            # Add summary section
            report_sections.append("## Executive Summary")
            report_sections.append(f"This diagnostic report provides a comprehensive analysis of {company_symbol} across financial, sustainability, and risk dimensions. The analysis is based on research-backed insights and quantitative metrics.")
            
            return '\n'.join(report_sections)
            
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {str(e)}")
            return f"Error generating diagnostic report for {company_symbol}: {str(e)}"
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context"""
        return {
            'history_length': len(self.conversation_history),
            'recent_entities': list(self.context_memory.keys())[-5:],  # Last 5 entities
            'recent_queries': [h['query_type'] for h in self.conversation_history[-3:]]  # Last 3 query types
        }
    
    def clear_conversation_context(self):
        """Clear conversation context"""
        self.conversation_history = []
        self.context_memory = {}
        self.logger.info("Conversation context cleared")
