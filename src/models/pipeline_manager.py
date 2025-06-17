#!/usr/bin/env python3
"""
Multi-stage AI Pipeline Manager for SPR Analysis

This module implements a sophisticated multi-stage pipeline that processes
company data through quality checks, AI analysis, and validation stages.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from models.quality_controller import DataQualityController, QualityMetrics
from models.hybrid_ai_model import HybridSPRModel
from models.drift_corrector import DriftCorrectionEngine, DriftDetectionResult, CorrectionResult
from models.rag_analyzer import RAGSPRAnalyzer, RAGAnalysisResult
from models.llm_interface import LLMInterface, LLMResponse


class PipelineStage(Enum):
    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing"
    QUALITY_CHECK = "quality_check"
    AI_ANALYSIS = "ai_analysis"
    VALIDATION = "validation"
    DRIFT_CORRECTION = "drift_correction"
    REPORT_GENERATION = "report_generation"


@dataclass
class PipelineResult:
    stage: PipelineStage
    success: bool
    data: Any
    quality_score: float
    processing_time: float
    confidence_score: float = 0.0
    error_message: str = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiStageSPRPipeline:
    """
    Advanced multi-stage pipeline for SPR analysis with quality control,
    drift detection, and hybrid AI models.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the multi-stage pipeline"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.quality_controller = DataQualityController(self.config)
        self.hybrid_model = HybridSPRModel(self.config)
        self.drift_corrector = DriftCorrectionEngine(self.config)
        self.rag_analyzer = RAGSPRAnalyzer(self.config)
        self.llm_interface = LLMInterface(self.config)
        
        # Pipeline configuration
        self.quality_thresholds = {
            'data_completeness': 0.8,
            'financial_accuracy': 0.9,
            'sustainability_relevance': 0.7,
            'overall_quality': 0.75
        }
        
        # Performance tracking
        self.pipeline_metrics = {
            'total_processed': 0,
            'success_rate': 0.0,
            'average_processing_time': 0.0,
            'stage_performance': {}
        }
          # Initialize async components flag
        self._components_initialized = False
    
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
    
    async def initialize_components(self):
        """Initialize async components (RAG and LLM systems)"""
        if not self._components_initialized:
            try:
                await self.rag_analyzer.initialize()
                await self.llm_interface.initialize()
                self._components_initialized = True
                self.logger.info("Pipeline components initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing pipeline components: {str(e)}")
                raise
    
    async def process_company(self, symbol: str, include_research: bool = True) -> Dict[str, Any]:
        """
        Execute the complete multi-stage pipeline for SPR analysis
        
        Args:
            symbol: Company stock symbol
            include_research: Whether to include research paper analysis
            
        Returns:
            Complete SPR analysis results with pipeline metrics
        """
        start_time = time.time()
        pipeline_results = {}
        
        try:
            self.logger.info(f"Starting multi-stage SPR pipeline for {symbol}")
            
            # Stage 1: Data Ingestion
            ingestion_result = await self._stage_data_ingestion(symbol, include_research)
            pipeline_results['ingestion'] = ingestion_result
            
            if not ingestion_result.success:
                return self._create_failure_response("Data ingestion failed", pipeline_results)
            
            # Stage 2: Preprocessing & Normalization
            preprocessing_result = await self._stage_preprocessing(ingestion_result.data, symbol)
            pipeline_results['preprocessing'] = preprocessing_result
            
            if not preprocessing_result.success:
                return self._create_failure_response("Preprocessing failed", pipeline_results)
            
            # Stage 3: Quality Control
            quality_result = await self._stage_quality_check(preprocessing_result.data, symbol)
            pipeline_results['quality_check'] = quality_result
            
            if not quality_result.success:
                return self._create_failure_response("Quality check failed", pipeline_results)
            
            # Stage 4: Drift Correction
            drift_result = await self._stage_drift_correction(quality_result.data, symbol)
            pipeline_results['drift_correction'] = drift_result
            
            # Stage 5: AI Analysis
            analysis_result = await self._stage_ai_analysis(drift_result.data, symbol)
            pipeline_results['analysis'] = analysis_result
            
            if not analysis_result.success:
                return self._create_failure_response("AI analysis failed", pipeline_results)
            
            # Stage 6: Validation
            validation_result = await self._stage_validation(analysis_result.data, symbol)
            pipeline_results['validation'] = validation_result
            
            # Stage 7: Report Generation
            final_result = await self._stage_report_generation(validation_result.data, pipeline_results)
            pipeline_results['final'] = final_result
            
            # Calculate overall pipeline quality
            pipeline_quality = self._calculate_pipeline_quality(pipeline_results)
            
            total_time = time.time() - start_time
            self._update_performance_metrics(pipeline_results, total_time)
            
            self.logger.info(f"Pipeline completed for {symbol} in {total_time:.2f}s")
            
            return {
                'symbol': symbol,
                'spr_results': final_result.data,
                'pipeline_quality': pipeline_quality,
                'processing_time': total_time,
                'pipeline_results': {k: asdict(v) for k, v in pipeline_results.items()},
                'confidence_score': final_result.confidence_score,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {symbol}: {str(e)}")
            return self._create_failure_response(f"Pipeline error: {str(e)}", pipeline_results)
    
    # ======= SPR Calculation Methods =======
    
    def _calculate_profit_performance_score(self, financial_metrics) -> float:
        """Calculate profit performance score (0-10 scale)"""
        try:
            # Handle both dict and object inputs
            if hasattr(financial_metrics, 'roi'):
                roi = financial_metrics.roi
                profit_margin = financial_metrics.profit_margin
                profitability_score = getattr(financial_metrics, 'profitability_score', 5.0)
                efficiency_score = getattr(financial_metrics, 'efficiency_score', 5.0)
            else:
                roi = financial_metrics.get('roi', 0)
                profit_margin = financial_metrics.get('profit_margin', 0)
                profitability_score = financial_metrics.get('profitability_score', 5.0)
                efficiency_score = financial_metrics.get('efficiency_score', 5.0)
            
            # Combine multiple profitability indicators
            scores = []
            
            # ROI score
            roi_score = max(0, min(10, (roi + 5) / 1.5))
            scores.append(roi_score * 0.3)
            
            # Profit margin score
            margin_score = max(0, min(10, (profit_margin + 10) / 2))
            scores.append(margin_score * 0.3)
            
            # Overall profitability score from financial processor
            scores.append(profitability_score * 0.2)
            
            # Efficiency score
            scores.append(efficiency_score * 0.2)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating profit performance score: {e}")
            return 5.0  # Default neutral score
    
    def _calculate_sustainability_impact_score(self, sustainability_metrics, research_insights) -> float:
        """Calculate sustainability impact score (0-10 scale)"""
        try:
            # Handle both dict and object inputs
            if hasattr(sustainability_metrics, 'esg_score'):
                esg_score = sustainability_metrics.esg_score
                environmental_score = sustainability_metrics.environmental_score
                social_score = sustainability_metrics.social_score
                governance_score = sustainability_metrics.governance_score
            else:
                esg_score = sustainability_metrics.get('esg_score', 5.0)
                environmental_score = sustainability_metrics.get('environmental_score', 5.0)
                social_score = sustainability_metrics.get('social_score', 5.0)
                governance_score = sustainability_metrics.get('governance_score', 5.0)
            
            scores = []
            
            # ESG score component (40%)
            if esg_score > 0:
                scores.append(min(10, esg_score) * 0.4)
            else:
                # Fallback: average of individual scores
                avg_score = (environmental_score + social_score + governance_score) / 3
                scores.append(min(10, avg_score) * 0.4)
            
            # Environmental component (25%)
            env_score = min(10, max(0, environmental_score))
            scores.append(env_score * 0.25)
            
            # Social component (20%)
            soc_score = min(10, max(0, social_score))
            scores.append(soc_score * 0.2)
            
            # Governance component (15%)
            gov_score = min(10, max(0, governance_score))
            scores.append(gov_score * 0.15)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating sustainability impact score: {e}")
            return 5.0
    
    def _calculate_research_alignment_score(self, research_insights) -> float:
        """Calculate research alignment score based on insights"""
        try:
            if not research_insights:
                return 0.0
            
            total_score = 0
            valid_insights = 0
            
            for insight in research_insights:
                if hasattr(insight, 'relevance_score'):
                    relevance = insight.relevance_score
                elif isinstance(insight, dict):
                    relevance = insight.get('relevance_score', 5.0)
                else:
                    relevance = 5.0
                
                # Weight by practice importance if available
                if hasattr(insight, 'practice_importance'):
                    importance = insight.practice_importance
                elif isinstance(insight, dict):
                    importance = insight.get('practice_importance', 1.0)
                else:
                    importance = 1.0
                
                weighted_score = relevance * importance
                total_score += weighted_score
                valid_insights += 1
            
            if valid_insights == 0:
                return 0.0
            
            # Average score, normalized to 0-10 scale
            avg_score = total_score / valid_insights
            return min(10, max(0, avg_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating research alignment score: {e}")
            return 0.0
    
    def _calculate_risk_factor(self, financial_metrics, sustainability_metrics) -> float:
        """Calculate risk factor (0-1 scale, where higher means more risk)"""
        try:
            risks = []
            
            # Financial risk indicators
            if hasattr(financial_metrics, 'debt_to_equity'):
                debt_ratio = financial_metrics.debt_to_equity
            else:
                debt_ratio = financial_metrics.get('debt_to_equity', 0.5)
            
            if debt_ratio > 2.0:
                risks.append(0.3)  # High debt risk
            elif debt_ratio > 1.0:
                risks.append(0.15)  # Moderate debt risk
            
            # ESG risk indicators
            if hasattr(sustainability_metrics, 'governance_score'):
                gov_score = sustainability_metrics.governance_score
            else:
                gov_score = sustainability_metrics.get('governance_score', 5.0)
            
            if gov_score < 3.0:
                risks.append(0.25)  # Poor governance risk
            elif gov_score < 5.0:
                risks.append(0.1)   # Moderate governance risk
            
            # Environmental risk
            if hasattr(sustainability_metrics, 'environmental_score'):
                env_score = sustainability_metrics.environmental_score
            else:
                env_score = sustainability_metrics.get('environmental_score', 5.0)
            
            if env_score < 3.0:
                risks.append(0.2)   # High environmental risk
            
            # Total risk factor (capped at 0.8)
            total_risk = min(0.8, sum(risks))
            return total_risk
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk factor: {e}")
            return 0.5
    
    def _calculate_spr_score(self, profit_score: float, sustainability_score: float, 
                           research_score: float, risk_factor: float) -> float:
        """Calculate the final SPR score"""
        try:
            # Use default weights if not configured
            weights = {
                'profit_performance': 0.4,
                'sustainability_impact': 0.35,
                'research_alignment': 0.25
            }
            
            # Weighted combination of components
            weighted_score = (
                profit_score * weights['profit_performance'] +
                sustainability_score * weights['sustainability_impact'] +
                research_score * weights['research_alignment']
            )
            
            # Adjust for risk (risk factor reduces the score)
            risk_adjusted_score = weighted_score * (1 - risk_factor * 0.2)
            
            # Ensure score is within 0-10 range
            return max(0, min(10, risk_adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating SPR score: {e}")
            return 5.0
    
    def _generate_spr_recommendations(self, spr_score: float, profit_score: float, 
                                    sustainability_score: float, research_score: float, 
                                    risk_factor: float) -> List[str]:
        """Generate actionable recommendations based on SPR analysis"""
        recommendations = []
        
        # Overall SPR recommendations
        if spr_score < 4:
            recommendations.append("Consider significant improvements in both profitability and sustainability practices")
        elif spr_score < 6:
            recommendations.append("Focus on enhancing weaker performance areas while maintaining strengths")
        elif spr_score < 8:
            recommendations.append("Fine-tune existing strategies for optimal sustainability-profit balance")
        else:
            recommendations.append("Maintain current excellent performance and consider leadership in industry sustainability")
        
        # Profit-specific recommendations
        if profit_score < 5:
            recommendations.append("Implement cost optimization and revenue enhancement strategies")
            recommendations.append("Review operational efficiency and market positioning")
        
        # Sustainability-specific recommendations
        if sustainability_score < 5:
            recommendations.append("Develop comprehensive ESG strategy and sustainability reporting")
            recommendations.append("Invest in environmental and social impact initiatives")
        
        # Research alignment recommendations
        if research_score < 3:
            recommendations.append("Align business practices with current sustainability research and best practices")
            recommendations.append("Consider partnerships with sustainability research institutions")
        
        # Risk mitigation recommendations
        if risk_factor > 0.6:
            recommendations.append("Address high-risk factors, particularly governance and financial stability")
            recommendations.append("Implement robust risk management and compliance frameworks")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _assess_data_completeness(self, data: Dict) -> float:
        """Assess completeness of ingested data"""
        required_fields = ['financial_metrics', 'sustainability_metrics']
        available = sum(1 for field in required_fields if field in data and data[field])
        return available / len(required_fields)
    
    def _normalize_financial_data(self, financial_data) -> Dict:
        """Normalize financial data to standard scales"""
        # Convert dataclass to dict if necessary
        if hasattr(financial_data, '__dict__'):
            # It's a dataclass or object, convert to dict
            if hasattr(financial_data, 'to_dict'):
                normalized = financial_data.to_dict()
            else:
                normalized = asdict(financial_data) if hasattr(financial_data, '__dataclass_fields__') else financial_data.__dict__.copy()
        else:
            # It's already a dict
            normalized = financial_data.copy()
        
        # Add normalization logic here (placeholder for now)
        return normalized
    
    def _normalize_sustainability_data(self, sustainability_data) -> Dict:
        """Normalize sustainability data to standard scales"""
        # Convert dataclass to dict if necessary
        if hasattr(sustainability_data, '__dict__'):
            # It's a dataclass or object, convert to dict
            if hasattr(sustainability_data, 'to_dict'):
                normalized = sustainability_data.to_dict()
            else:
                normalized = asdict(sustainability_data) if hasattr(sustainability_data, '__dataclass_fields__') else sustainability_data.__dict__.copy()
        else:
            # It's already a dict
            normalized = sustainability_data.copy()
        
        # Add normalization logic here (placeholder for now)
        return normalized
    
    def _extract_research_insights(self, research_data: Dict) -> List[Dict]:
        """Extract structured insights from research data"""
        insights = research_data.get('insights', [])
        return [asdict(insight) for insight in insights]
    
    def _assess_preprocessing_quality(self, data: Dict) -> float:
        """Assess quality of preprocessing stage"""
        quality_factors = []
        
        # Check data completeness after preprocessing
        if 'financial_metrics' in data:
            quality_factors.append(0.9)
        if 'sustainability_metrics' in data:
            quality_factors.append(0.9)
        if 'research_insights' in data:
            quality_factors.append(0.8)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    # ======= Pipeline Stage Methods =======
    
    async def _stage_data_ingestion(self, symbol: str, include_research: bool) -> PipelineResult:
        """Stage 1: Collect and validate raw data from multiple sources"""
        start_time = time.time()
        
        try:
            from financial.data_processor import FinancialDataProcessor
            from research_processor.paper_analyzer import ResearchAnalyzer
            
            # Initialize data processors
            financial_processor = FinancialDataProcessor()
            research_analyzer = ResearchAnalyzer() if include_research else None
            
            # Collect financial data
            financial_metrics = await financial_processor.get_financial_metrics(symbol)
            sustainability_metrics = await financial_processor.get_sustainability_metrics(symbol)
            
            # Collect research data if requested
            research_data = None
            if include_research and research_analyzer:
                research_query = f"sustainability {symbol} corporate performance profitability"
                research_results = await research_analyzer.analyze_sustainability_papers(
                    research_query, max_papers=20
                )
                research_data = research_results
            
            # Compile raw data
            raw_data = {
                'financial_metrics': financial_metrics,
                'sustainability_metrics': sustainability_metrics,
                'research_data': research_data,
                'symbol': symbol,
                'ingestion_timestamp': time.time()
            }
            
            # Basic validation
            data_completeness = self._assess_data_completeness(raw_data)
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.DATA_INGESTION,
                success=data_completeness > 0.5,
                data=raw_data,
                quality_score=data_completeness,
                processing_time=processing_time,
                confidence_score=data_completeness,
                metadata={'completeness': data_completeness}
            )
            
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DATA_INGESTION,
                success=False,
                data={},
                quality_score=0.0,
                processing_time=time.time() - start_time,                error_message=str(e)
            )
    
    async def _stage_preprocessing(self, raw_data: Dict, symbol: str) -> PipelineResult:
        """Stage 2: Clean, normalize, and prepare data for analysis"""
        start_time = time.time()
        
        try:
            processed_data = raw_data.copy()
            
            # Normalize financial metrics
            if 'financial_metrics' in processed_data:
                self.logger.info(f"Normalizing financial data for {symbol}")
                processed_data['financial_metrics'] = self._normalize_financial_data(
                    processed_data['financial_metrics']
                )
            
            # Normalize sustainability metrics
            if 'sustainability_metrics' in processed_data:
                self.logger.info(f"Normalizing sustainability data for {symbol}")
                processed_data['sustainability_metrics'] = self._normalize_sustainability_data(
                    processed_data['sustainability_metrics']
                )
            
            # Process research data
            if 'research_data' in processed_data and processed_data['research_data']:
                self.logger.info(f"Extracting research insights for {symbol}")
                processed_data['research_insights'] = self._extract_research_insights(
                    processed_data['research_data']
                )
            
            # Calculate preprocessing quality
            preprocessing_quality = self._assess_preprocessing_quality(processed_data)
            self.logger.info(f"Preprocessing quality for {symbol}: {preprocessing_quality}")
            
            processing_time = time.time() - start_time
            
            success = preprocessing_quality > 0.6
            self.logger.info(f"Preprocessing {'succeeded' if success else 'failed'} for {symbol} (quality: {preprocessing_quality})")
            
            return PipelineResult(
                stage=PipelineStage.PREPROCESSING,
                success=success,
                data=processed_data,
                quality_score=preprocessing_quality,
                processing_time=processing_time,
                confidence_score=preprocessing_quality
            )
            
        except Exception as e:
            self.logger.error(f"Preprocessing error for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return PipelineResult(
                stage=PipelineStage.PREPROCESSING,
                success=False,
                data=raw_data,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _stage_quality_check(self, processed_data: Dict, symbol: str) -> PipelineResult:
        """Stage 3: Comprehensive quality assessment and filtering"""
        start_time = time.time()
        
        try:
            # Assess financial data quality
            financial_quality = self.quality_controller.assess_financial_data_quality(
                processed_data.get('financial_metrics', {})
            )
            
            # Assess sustainability data quality
            sustainability_quality = self.quality_controller.assess_sustainability_data_quality(
                processed_data.get('sustainability_metrics', {})
            )
            
            # Calculate overall quality
            overall_quality = (financial_quality.overall_quality + sustainability_quality.overall_quality) / 2
            
            # Filter low-quality data
            filtered_data, quality_passed = self.quality_controller.filter_low_quality_inputs(
                processed_data, min_quality=self.quality_thresholds['overall_quality']
            )
            
            if quality_passed:
                result_data = filtered_data
            else:
                result_data = processed_data  # Keep original but flag quality issues
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.QUALITY_CHECK,
                success=quality_passed,
                data=result_data,
                quality_score=overall_quality,
                processing_time=processing_time,
                confidence_score=overall_quality,
                metadata={
                    'financial_quality': asdict(financial_quality),
                    'sustainability_quality': asdict(sustainability_quality),
                    'quality_passed': quality_passed
                }
            )
            
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.QUALITY_CHECK,
                success=False,
                data=processed_data,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _stage_drift_correction(self, quality_data: Dict, symbol: str) -> PipelineResult:
        """Stage 4: Detect and correct data drift"""
        start_time = time.time()
        
        try:
            corrected_data = quality_data.copy()
            drift_corrections = {}
            
            # Check for financial data drift
            if 'financial_metrics' in corrected_data:
                financial_drift = self.drift_corrector.detect_financial_drift(
                    corrected_data['financial_metrics'], symbol
                )
                if financial_drift['correction_needed']:
                    corrected_data['financial_metrics'] = self.drift_corrector.correct_financial_drift(
                        corrected_data['financial_metrics'], symbol
                    )
                    drift_corrections['financial'] = financial_drift
            
            # Check for sustainability methodology drift
            if 'sustainability_metrics' in corrected_data:
                sustainability_drift = self.drift_corrector.detect_sustainability_drift(
                    corrected_data['sustainability_metrics'], symbol
                )
                if sustainability_drift['correction_needed']:
                    corrected_data['sustainability_metrics'] = self.drift_corrector.correct_sustainability_drift(
                        corrected_data['sustainability_metrics'], symbol
                    )
                    drift_corrections['sustainability'] = sustainability_drift
            
            drift_quality = 1.0 - len(drift_corrections) * 0.1  # Slight penalty for corrections
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.DRIFT_CORRECTION,
                success=True,
                data=corrected_data,
                quality_score=drift_quality,
                processing_time=processing_time,
                confidence_score=drift_quality,
                metadata={'drift_corrections': drift_corrections}
            )
            
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DRIFT_CORRECTION,
                success=False,
                data=quality_data,
                quality_score=0.8,  # Default quality if drift correction fails
                processing_time=time.time() - start_time,
                error_message=str(e)            )
    
    async def _stage_ai_analysis(self, drift_corrected_data: Dict, symbol: str) -> PipelineResult:
        """Stage 5: Advanced AI analysis using hybrid models and SPR calculations"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting AI analysis and SPR calculation for {symbol}")
            
            # Extract data for calculations
            financial_metrics = drift_corrected_data.get('financial_metrics', {})
            sustainability_metrics = drift_corrected_data.get('sustainability_metrics', {})
            research_insights = drift_corrected_data.get('research_insights', [])
            
            # Calculate component scores using the original SPR logic
            profit_score = self._calculate_profit_performance_score(financial_metrics)
            sustainability_score = self._calculate_sustainability_impact_score(
                sustainability_metrics, research_insights
            )
            research_alignment_score = self._calculate_research_alignment_score(research_insights)
            risk_factor = self._calculate_risk_factor(financial_metrics, sustainability_metrics)
            
            # Calculate final SPR score
            spr_score = self._calculate_spr_score(
                profit_score, sustainability_score, research_alignment_score, risk_factor
            )
            
            self.logger.info(f"SPR Calculation for {symbol}:")
            self.logger.info(f"  - Profit Performance: {profit_score:.2f}")
            self.logger.info(f"  - Sustainability Impact: {sustainability_score:.2f}")
            self.logger.info(f"  - Research Alignment: {research_alignment_score:.2f}")
            self.logger.info(f"  - Risk Factor: {risk_factor:.2f}")
            self.logger.info(f"  - Final SPR Score: {spr_score:.2f}")
            
            # Try to get additional insights from hybrid model (non-blocking)
            ai_insights = []
            ai_confidence = 0.8
            try:
                # Prepare data for hybrid model
                financial_features = self._prepare_financial_features(financial_metrics)
                sustainability_features = self._prepare_sustainability_features(
                    sustainability_metrics, research_insights
                )
                
                # Run hybrid AI analysis for additional insights
                ai_results = await self.hybrid_model.analyze(
                    financial_features=financial_features,
                    sustainability_features=sustainability_features,
                    symbol=symbol
                )
                
                ai_insights = ai_results.get('insights', [])
                ai_confidence = ai_results.get('confidence_score', 0.8)
                
            except Exception as ai_error:
                self.logger.warning(f"Hybrid AI model analysis failed for {symbol}: {ai_error}")
                # Continue with traditional calculations
                
            # Compile analysis results
            analysis_results = {
                'spr_score': spr_score,
                'profit_performance_score': profit_score,
                'sustainability_impact_score': sustainability_score,
                'research_alignment_score': research_alignment_score,
                'risk_factor': risk_factor,
                'ai_insights': ai_insights,
                'company_name': symbol,  # Will be updated with real name if available
                'confidence_score': ai_confidence,
                'key_findings': [
                    f"SPR Score: {spr_score:.2f}/10",
                    f"Profit Performance: {profit_score:.2f}/10",
                    f"Sustainability Impact: {sustainability_score:.2f}/10",
                    f"Research Alignment: {research_alignment_score:.2f}/10"
                ],
                'recommendations': self._generate_spr_recommendations(
                    spr_score, profit_score, sustainability_score, research_alignment_score, risk_factor
                )
            }
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.AI_ANALYSIS,
                success=True,
                data=analysis_results,
                quality_score=0.9,  # High quality for successful calculation
                processing_time=processing_time,
                confidence_score=ai_confidence,
                metadata={
                    'calculation_method': 'traditional_spr_with_ai_enhancement',
                    'component_scores': {
                        'profit': profit_score,
                        'sustainability': sustainability_score,
                        'research': research_alignment_score,
                        'risk': risk_factor
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in AI analysis for {symbol}: {str(e)}")
            # Fallback to basic calculations
            fallback_results = {
                'spr_score': 5.0,
                'profit_performance_score': 5.0,
                'sustainability_impact_score': 5.0,
                'research_alignment_score': 0.0,
                'risk_factor': 0.5,
                'company_name': symbol,
                'key_findings': ['Analysis failed, using default values'],
                'recommendations': ['Unable to generate recommendations due to analysis failure']
            }
            
            return PipelineResult(
                stage=PipelineStage.AI_ANALYSIS,
                success=False,
                data=fallback_results,
                quality_score=0.6,  # Lower quality for fallback
                processing_time=time.time() - start_time,
                confidence_score=0.4,
                error_message=f"AI analysis failed: {str(e)}"
            )
            
        except Exception as e:
            # Fallback to traditional calculation
            traditional_results = self._calculate_fallback_spr(drift_corrected_data)
            
            return PipelineResult(
                stage=PipelineStage.AI_ANALYSIS,
                success=False,
                data=traditional_results,
                quality_score=0.6,  # Lower quality for fallback
                processing_time=time.time() - start_time,
                confidence_score=0.6,
                error_message=f"AI analysis failed, using fallback: {str(e)}"
            )
    
    async def _stage_validation(self, analysis_data: Dict, symbol: str) -> PipelineResult:
        """Stage 6: Validate results against benchmarks and sanity checks"""
        start_time = time.time()
        
        try:
            validation_results = {}
            
            # Sanity checks
            spr_score = analysis_data.get('spr_score', 0)
            validation_results['sanity_checks'] = {
                'spr_in_range': 0 <= spr_score <= 10,
                'components_balanced': self._check_component_balance(analysis_data),
                'no_extreme_outliers': self._check_for_outliers(analysis_data)
            }
            
            # Industry benchmarking
            industry_benchmark = await self._get_industry_benchmark(symbol)
            if industry_benchmark:
                validation_results['industry_comparison'] = {
                    'relative_performance': spr_score / industry_benchmark.get('average_spr', 5.0),
                    'industry_percentile': self._calculate_industry_percentile(spr_score, industry_benchmark)
                }
            
            # Confidence assessment
            confidence_factors = [
                validation_results['sanity_checks']['spr_in_range'],
                validation_results['sanity_checks']['components_balanced'],
                validation_results['sanity_checks']['no_extreme_outliers'],
            ]
            
            validation_confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Add validation metadata to results
            validated_data = analysis_data.copy()
            validated_data['validation'] = validation_results
            validated_data['validation_confidence'] = validation_confidence
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.VALIDATION,
                success=validation_confidence > 0.7,
                data=validated_data,
                quality_score=validation_confidence,
                processing_time=processing_time,
                confidence_score=validation_confidence,
                metadata=validation_results
            )
            
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.VALIDATION,
                success=False,
                data=analysis_data,
                quality_score=0.5,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _stage_report_generation(self, validated_data: Dict, pipeline_results: Dict) -> PipelineResult:
        """Stage 7: Generate comprehensive SPR report with explanations"""
        start_time = time.time()
        
        try:
            # Generate comprehensive report
            report = {
                'spr_analysis': validated_data,
                'executive_summary': self._generate_executive_summary(validated_data),
                'detailed_breakdown': self._generate_detailed_breakdown(validated_data),
                'recommendations': self._generate_recommendations(validated_data),
                'risk_assessment': self._generate_risk_assessment(validated_data),
                'pipeline_metadata': {
                    'processing_quality': self._calculate_pipeline_quality(pipeline_results),
                    'confidence_score': validated_data.get('validation_confidence', 0.8),
                    'data_sources': self._extract_data_sources(pipeline_results),
                    'processing_stages': list(pipeline_results.keys())
                }
            }
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.REPORT_GENERATION,
                success=True,
                data=report,
                quality_score=0.95,
                processing_time=processing_time,
                confidence_score=validated_data.get('validation_confidence', 0.8)
            )
            
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.REPORT_GENERATION,
                success=False,
                data=validated_data,
                quality_score=0.7,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _calculate_pipeline_quality(self, pipeline_results: Dict) -> float:
        """Calculate overall pipeline quality score"""
        quality_scores = []
        
        for stage_name, result in pipeline_results.items():
            if isinstance(result, PipelineResult):
                quality_scores.append(result.quality_score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _extract_data_sources(self, pipeline_results: Dict) -> List[str]:
        """Extract data sources used in the pipeline"""
        sources = ['Financial APIs', 'Sustainability Databases']
        
        if 'ingestion' in pipeline_results:
            ingestion_data = pipeline_results['ingestion'].data
            if ingestion_data.get('research_data'):
                sources.append('Academic Research Papers')
        
        return sources
    
    def _update_performance_metrics(self, pipeline_results: Dict, total_time: float):
        """Update pipeline performance metrics"""
        self.pipeline_metrics['total_processed'] += 1
        
        # Update average processing time
        current_avg = self.pipeline_metrics['average_processing_time']
        total_processed = self.pipeline_metrics['total_processed']
        self.pipeline_metrics['average_processing_time'] = (
            (current_avg * (total_processed - 1) + total_time) / total_processed
        )
        
        # Update success rate
        success = all(result.success for result in pipeline_results.values() 
                     if isinstance(result, PipelineResult))
        current_success_rate = self.pipeline_metrics['success_rate']
        self.pipeline_metrics['success_rate'] = (
            (current_success_rate * (total_processed - 1) + (1 if success else 0)) / total_processed
        )
    
    def _create_failure_response(self, error_message: str, pipeline_results: Dict) -> Dict[str, Any]:
        """Create a standardized failure response"""
        return {
            'success': False,
            'error': error_message,
            'pipeline_results': {k: asdict(v) for k, v in pipeline_results.items()},
            'spr_results': None,
            'pipeline_quality': self._calculate_pipeline_quality(pipeline_results)
        }
    
    def get_pipeline_performance(self) -> Dict[str, Any]:
        """Get current pipeline performance metrics"""
        return self.pipeline_metrics.copy()
