"""
Main Sustainability Profit Ratio (SPR) Analyzer

This module combines research insights with financial data to calculate
the Sustainability Profit Ratio and provide comprehensive analysis.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
import json
import sqlite3
import os

from research_processor.paper_analyzer import ResearchAnalyzer, SustainabilityInsight
from financial.data_processor import FinancialDataProcessor, FinancialMetrics, SustainabilityMetrics
from utils.config_loader import ConfigLoader
from models.pipeline_manager import MultiStageSPRPipeline


@dataclass
class SPRResult:
    """Data class for SPR analysis results"""
    symbol: str
    company_name: str
    spr_score: float
    
    # Component scores
    profit_performance_score: float
    sustainability_impact_score: float
    research_alignment_score: float
    risk_factor: float
    
    # Raw metrics
    financial_metrics: FinancialMetrics
    sustainability_metrics: SustainabilityMetrics
    research_insights: List[SustainabilityInsight]
      # Analysis metadata
    analysis_date: datetime
    confidence_level: float
    key_findings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to string
        result['analysis_date'] = self.analysis_date.isoformat()
        
        # Handle nested objects that might contain datetime or other non-serializable objects
        if hasattr(self.financial_metrics, 'to_dict'):
            result['financial_metrics'] = self.financial_metrics.to_dict()
        if hasattr(self.sustainability_metrics, 'to_dict'):
            result['sustainability_metrics'] = self.sustainability_metrics.to_dict()
            
        # Convert research insights to dicts if they have the method
        if self.research_insights:
            result['research_insights'] = []
            for insight in self.research_insights:
                if hasattr(insight, 'to_dict'):
                    result['research_insights'].append(insight.to_dict())
                else:
                    result['research_insights'].append(str(insight))
                    
        return result


class SPRAnalyzer:
    """
    Main Sustainability Profit Ratio Analyzer
    
    Combines research insights with financial data to calculate SPR scores
    and provide comprehensive sustainability-profitability analysis.    """
    
    def __init__(self, config_path: str = None):
        """Initialize the SPR analyzer"""
        self.config = ConfigLoader(config_path).config
        self.logger = self._setup_logging()
        
        # Initialize component analyzers
        self.research_analyzer = ResearchAnalyzer(config_path)
        self.financial_processor = FinancialDataProcessor(config_path)
        
        # Initialize advanced multi-stage pipeline
        self.pipeline = MultiStageSPRPipeline(config_path)
        self._pipeline_initialized = False
        
        # Initialize database
        self._setup_database()
        
        # SPR calculation weights
        self.weights = self.config['spr_calculation']['weights']
        
        self.logger.info("SPR Analyzer initialized successfully")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create logs directory
            log_dir = os.path.dirname(self.config['logging']['file'])
            os.makedirs(log_dir, exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(self.config['logging']['file'])
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(self.config['logging']['format'])
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
        
    def _setup_database(self):
        """Set up SQLite database for caching and storage"""
        try:
            db_path = self.config['data']['storage']['database']
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            
            # Create tables
            self._create_database_tables()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise
            
    def _create_database_tables(self):
        """Create necessary database tables"""
        cursor = self.db_connection.cursor()
        
        # SPR results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT,
                spr_score REAL,
                profit_performance_score REAL,
                sustainability_impact_score REAL,
                research_alignment_score REAL,
                risk_factor REAL,
                analysis_date TEXT,
                confidence_level REAL,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Research insights cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                query TEXT,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        # Financial data cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                data_type TEXT,
                data_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        self.db_connection.commit()
        
    async def analyze_company_advanced(self, symbol: str, include_research: bool = True) -> Optional[SPRResult]:
        """
        Perform comprehensive SPR analysis using the advanced multi-stage pipeline
        
        This method uses the new pipeline with quality control, drift correction,
        hybrid AI models, RAG analysis, and LLM-powered insights.
        
        Args:
            symbol: Stock symbol (e.g., 'TSLA')
            include_research: Whether to include research paper analysis
            
        Returns:
            SPRResult with comprehensive analysis or None if analysis fails
        """
        try:
            self.logger.info(f"Starting advanced SPR analysis for {symbol}")
            
            # Initialize pipeline components if needed
            if not self._pipeline_initialized:
                await self.pipeline.initialize_components()
                self._pipeline_initialized = True
            
            # Run the multi-stage pipeline
            pipeline_result = await self.pipeline.process_company(symbol, include_research)
            
            if not pipeline_result['success']:
                self.logger.error(f"Pipeline failed for {symbol}: {pipeline_result.get('error', 'Unknown error')}")
                return None
            
            # Extract SPR results from pipeline
            spr_data = pipeline_result['spr_results']
            
            # Convert to SPRResult format
            result = SPRResult(
                symbol=symbol,
                company_name=spr_data.get('company_name', symbol),
                spr_score=spr_data.get('spr_score', 0.0),
                profit_performance_score=spr_data.get('profit_performance_score', 0.0),
                sustainability_impact_score=spr_data.get('sustainability_impact_score', 0.0),
                research_alignment_score=spr_data.get('research_alignment_score', 0.0),
                risk_factor=spr_data.get('risk_factor', 1.0),
                financial_metrics=spr_data.get('financial_metrics', {}),
                sustainability_metrics=spr_data.get('sustainability_metrics', {}),
                research_insights=spr_data.get('research_insights', []),
                analysis_date=datetime.now(),
                confidence_level=pipeline_result.get('confidence_score', 0.0),
                key_findings=spr_data.get('key_findings', []),
                recommendations=spr_data.get('recommendations', [])            )
            
            # Log successful completion (caching can be added later if needed)
            
            self.logger.info(f"Advanced SPR analysis completed for {symbol}")
            self.logger.info(f"SPR Score: {result.spr_score:.2f}")
            self.logger.info(f"Research Alignment: {result.research_alignment_score:.2f}")
            self.logger.info(f"Pipeline Quality: {pipeline_result['pipeline_quality']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced SPR analysis for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    async def analyze_company(self, symbol: str, include_research: bool = True) -> Optional[SPRResult]:
        """
        Perform comprehensive SPR analysis for a company
        
        Args:
            symbol: Stock symbol (e.g., 'TSLA')
            include_research: Whether to include research paper analysis
            
        Returns:
            SPRResult object with complete analysis
        """
        try:
            self.logger.info(f"Starting SPR analysis for {symbol}")
            
            # Step 1: Get financial metrics
            financial_metrics = await self.financial_processor.get_financial_metrics(symbol)
            if not financial_metrics:
                self.logger.error(f"Could not retrieve financial data for {symbol}")
                return None
                
            # Step 2: Get sustainability metrics
            sustainability_metrics = await self.financial_processor.get_sustainability_metrics(symbol)
            if not sustainability_metrics:
                self.logger.warning(f"Limited sustainability data for {symbol}")
                sustainability_metrics = SustainabilityMetrics(symbol=symbol)
                
            # Step 3: Get research insights (if requested)
            research_insights = []
            if include_research:
                research_query = f"sustainability {financial_metrics.company_name} corporate performance profitability"
                research_results = await self.research_analyzer.analyze_sustainability_papers(
                    research_query, max_papers=20
                )
                research_insights = research_results.get('insights', [])
                
            # Step 4: Calculate SPR components
            profit_score = self._calculate_profit_performance_score(financial_metrics)
            sustainability_score = self._calculate_sustainability_impact_score(
                sustainability_metrics, research_insights
            )
            research_alignment_score = self._calculate_research_alignment_score(research_insights)
            risk_factor = self._calculate_risk_factor(financial_metrics, sustainability_metrics)
            
            # Step 5: Calculate final SPR score
            spr_score = self._calculate_spr_score(
                profit_score, sustainability_score, research_alignment_score, risk_factor
            )
            
            # Step 6: Generate insights and recommendations
            key_findings = self._generate_key_findings(
                financial_metrics, sustainability_metrics, research_insights
            )
            recommendations = self._generate_recommendations(
                financial_metrics, sustainability_metrics, research_insights
            )
            
            # Step 7: Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                financial_metrics, sustainability_metrics, research_insights
            )
            
            # Create result object
            result = SPRResult(
                symbol=symbol,
                company_name=financial_metrics.company_name,
                spr_score=spr_score,
                profit_performance_score=profit_score,
                sustainability_impact_score=sustainability_score,
                research_alignment_score=research_alignment_score,
                risk_factor=risk_factor,
                financial_metrics=financial_metrics,
                sustainability_metrics=sustainability_metrics,
                research_insights=research_insights,
                analysis_date=datetime.now(),
                confidence_level=confidence_level,
                key_findings=key_findings,
                recommendations=recommendations
            )
              # Save to database
            self._save_spr_result(result)
            
            self.logger.info(f"SPR analysis completed for {symbol}: SPR = {spr_score:.2f}")
            
            # Return dictionary format for API compatibility
            return {
                'symbol': symbol,
                'company': financial_metrics.company_name,
                'spr_score': spr_score,
                'profit_performance': profit_score,
                'sustainability_impact': sustainability_score,
                'research_alignment': research_alignment_score,
                'risk_factor': risk_factor,
                'confidence_level': confidence_level,
                'key_findings': key_findings,
                'recommendations': recommendations,
                'analysis_date': datetime.now().isoformat(),
                'financial_metrics': {
                    'revenue': financial_metrics.revenue,
                    'net_income': financial_metrics.net_income,
                    'roi': financial_metrics.roi,
                    'profit_margin': financial_metrics.profit_margin,
                    'profitability_score': financial_metrics.profitability_score
                },                'sustainability_metrics': {
                    'esg_score': sustainability_metrics.esg_score,
                    'environmental_score': sustainability_metrics.environmental_score,
                    'social_score': sustainability_metrics.social_score,
                    'governance_score': sustainability_metrics.governance_score,
                    'carbon_intensity': sustainability_metrics.carbon_intensity,
                    'renewable_energy_percentage': sustainability_metrics.renewable_energy_percentage
                },
                'research_insights_count': len(research_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Error in SPR analysis for {symbol}: {e}")
            return None
            
    def _calculate_profit_performance_score(self, financial_metrics: FinancialMetrics) -> float:
        """Calculate profit performance score (0-10 scale)"""
        try:
            # Combine multiple profitability indicators
            scores = []
            
            # ROI score
            roi_score = max(0, min(10, (financial_metrics.roi + 5) / 1.5))
            scores.append(roi_score * 0.3)
            
            # Profit margin score
            margin_score = max(0, min(10, (financial_metrics.profit_margin + 10) / 2))
            scores.append(margin_score * 0.3)
            
            # Overall profitability score from financial processor
            scores.append(financial_metrics.profitability_score * 0.2)
            
            # Efficiency score
            scores.append(financial_metrics.efficiency_score * 0.2)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating profit performance score: {e}")
            return 5.0  # Default neutral score
            
    def _calculate_sustainability_impact_score(self, 
                                             sustainability_metrics: SustainabilityMetrics,
                                             research_insights: List[SustainabilityInsight]) -> float:
        """Calculate sustainability impact score (0-10 scale)"""
        try:
            scores = []
            
            # ESG score component
            if sustainability_metrics.esg_score > 0:
                scores.append(sustainability_metrics.esg_score * 0.4)
                
            # Environmental score component
            if sustainability_metrics.environmental_score > 0:
                scores.append(sustainability_metrics.environmental_score * 0.3)
                
            # Research-backed sustainability impact
            if research_insights:
                avg_research_impact = np.mean([
                    insight.financial_correlation * insight.confidence_score 
                    for insight in research_insights
                ])
                research_score = max(0, min(10, avg_research_impact * 10))
                scores.append(research_score * 0.3)
                
            if not scores:
                return 5.0  # Default neutral score
                
            return sum(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating sustainability impact score: {e}")
            return 5.0
            
    def _calculate_research_alignment_score(self, research_insights: List[SustainabilityInsight]) -> float:
        """Calculate how well the company aligns with research-backed practices"""
        try:
            if not research_insights:
                return 5.0  # Neutral score if no research data
                
            # Weight by confidence and financial correlation
            weighted_scores = []
            for insight in research_insights:
                score = insight.financial_correlation * insight.confidence_score * 10
                weighted_scores.append(score)
                
            return np.mean(weighted_scores) if weighted_scores else 5.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating research alignment score: {e}")
            return 5.0
            
    def _calculate_risk_factor(self, 
                             financial_metrics: FinancialMetrics,
                             sustainability_metrics: SustainabilityMetrics) -> float:
        """Calculate risk factor (lower is better)"""
        try:
            risk_components = []
            
            # Financial risk (debt-to-equity)
            if financial_metrics.debt_to_equity > 0:
                debt_risk = min(1.0, financial_metrics.debt_to_equity / 2.0)
                risk_components.append(debt_risk * 0.4)
                
            # Sustainability risk (inverse of ESG score)
            if sustainability_metrics.esg_score > 0:
                sustainability_risk = (10 - sustainability_metrics.esg_score) / 10
                risk_components.append(sustainability_risk * 0.3)
                
            # Market volatility risk (placeholder)
            market_risk = 0.3  # Would be calculated from actual volatility data
            risk_components.append(market_risk * 0.3)
            
            return sum(risk_components) if risk_components else 0.5
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk factor: {e}")
            return 0.5
            
    def _calculate_spr_score(self, 
                           profit_score: float,
                           sustainability_score: float,
                           research_score: float,
                           risk_factor: float) -> float:
        """Calculate the final SPR score"""
        try:
            # Weighted combination of components
            weighted_score = (
                profit_score * self.weights['profit_performance'] +
                sustainability_score * self.weights['sustainability_impact'] +
                research_score * self.weights['research_alignment']
            )
            
            # Adjust for risk (risk factor reduces the score)
            risk_adjusted_score = weighted_score * (1 - risk_factor * 0.2)
            
            # Ensure score is within 0-10 range
            return max(0, min(10, risk_adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating SPR score: {e}")
            return 5.0
            
    def _generate_key_findings(self,
                             financial_metrics: FinancialMetrics,
                             sustainability_metrics: SustainabilityMetrics,
                             research_insights: List[SustainabilityInsight]) -> List[str]:
        """Generate key findings from the analysis"""
        findings = []
        
        try:
            # Financial performance findings
            if financial_metrics.profit_margin > 10:
                findings.append(f"Strong profitability with {financial_metrics.profit_margin:.1f}% profit margin")
            elif financial_metrics.profit_margin < 0:
                findings.append("Company is currently operating at a loss")
                
            # ROI findings
            if financial_metrics.roi > 15:
                findings.append(f"Excellent return on investment at {financial_metrics.roi:.1f}%")
            elif financial_metrics.roi < 5:
                findings.append("Below-average return on investment")
                
            # Sustainability findings
            if sustainability_metrics.esg_score > 8:
                findings.append("High ESG performance indicating strong sustainability practices")
            elif sustainability_metrics.esg_score < 5:
                findings.append("ESG performance below industry standards")
                
            # Research insights findings
            high_impact_insights = [i for i in research_insights if i.financial_correlation > 0.7]
            if high_impact_insights:
                top_practice = high_impact_insights[0].practice
                findings.append(f"Research shows {top_practice} has strong correlation with financial performance")
                
            # Debt findings
            if financial_metrics.debt_to_equity > 1.5:
                findings.append("High debt levels may pose financial risk")
            elif financial_metrics.debt_to_equity < 0.3:
                findings.append("Conservative debt management with low leverage")
                
        except Exception as e:
            self.logger.warning(f"Error generating key findings: {e}")
            
        return findings[:5]  # Limit to top 5 findings
        
    def _generate_recommendations(self,
                                financial_metrics: FinancialMetrics,
                                sustainability_metrics: SustainabilityMetrics,
                                research_insights: List[SustainabilityInsight]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Profitability recommendations
            if financial_metrics.profit_margin < 5:
                recommendations.append("Focus on cost optimization and operational efficiency improvements")
                
            # Sustainability recommendations
            if sustainability_metrics.esg_score < 7:
                recommendations.append("Invest in ESG initiatives to improve sustainability profile")
                
            # Research-based recommendations
            for insight in research_insights[:3]:  # Top 3 insights
                if insight.financial_correlation > 0.6:
                    recommendations.append(
                        f"Consider investing in {insight.practice.lower()} based on research evidence"
                    )
                    
            # Risk management recommendations
            if financial_metrics.debt_to_equity > 1.0:
                recommendations.append("Consider debt reduction strategies to improve financial stability")
                
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Continue monitoring sustainability and financial performance metrics")
                
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
            
        return recommendations[:5]  # Limit to top 5 recommendations
        
    def _calculate_confidence_level(self,
                                  financial_metrics: FinancialMetrics,
                                  sustainability_metrics: SustainabilityMetrics,
                                  research_insights: List[SustainabilityInsight]) -> float:
        """Calculate confidence level of the analysis (0-1 scale)"""
        try:
            confidence_factors = []
            
            # Financial data completeness
            financial_completeness = 0.0
            if financial_metrics.revenue > 0:
                financial_completeness += 0.25
            if financial_metrics.net_income != 0:
                financial_completeness += 0.25
            if financial_metrics.total_assets > 0:
                financial_completeness += 0.25
            if financial_metrics.market_cap > 0:
                financial_completeness += 0.25
                
            confidence_factors.append(financial_completeness * 0.4)
            
            # Sustainability data quality
            sustainability_completeness = 0.0
            if sustainability_metrics.esg_score > 0:
                sustainability_completeness += 0.5
            if sustainability_metrics.environmental_score > 0:
                sustainability_completeness += 0.25
            if sustainability_metrics.social_score > 0:
                sustainability_completeness += 0.25
                
            confidence_factors.append(sustainability_completeness * 0.3)
              # Research data quality
            research_quality = 0.0
            if research_insights:
                avg_confidence = np.mean([i.confidence_score for i in research_insights])
                research_quality = avg_confidence
                
            confidence_factors.append(research_quality * 0.3)
            
            return sum(confidence_factors)
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence level: {e}")
            return 0.5
            
    def _save_spr_result(self, result: SPRResult):
        """Save SPR result to database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Create a simplified dict for JSON storage that avoids datetime serialization issues
            simple_dict = {
                'symbol': result.symbol,
                'company_name': result.company_name,
                'spr_score': result.spr_score,
                'profit_performance_score': result.profit_performance_score,
                'sustainability_impact_score': result.sustainability_impact_score,
                'research_alignment_score': result.research_alignment_score,
                'risk_factor': result.risk_factor,
                'confidence_level': result.confidence_level,
                'key_findings': result.key_findings,
                'recommendations': result.recommendations,
                'analysis_date': result.analysis_date.isoformat(),
                'research_insights_count': len(result.research_insights)
            }
            
            cursor.execute('''
                INSERT INTO spr_results (
                    symbol, company_name, spr_score, profit_performance_score,
                    sustainability_impact_score, research_alignment_score,
                    risk_factor, analysis_date, confidence_level, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.symbol,
                result.company_name,
                result.spr_score,
                result.profit_performance_score,
                result.sustainability_impact_score,
                result.research_alignment_score,
                result.risk_factor,
                result.analysis_date.isoformat(),
                result.confidence_level,
                json.dumps(simple_dict)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.warning(f"Error saving SPR result: {e}")
            
    async def compare_companies(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Compare SPR scores across multiple companies
        
        Args:
            symbols: List of stock symbols to compare
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"Comparing companies: {symbols}")
            
            # Analyze each company
            results = {}
            for symbol in symbols:
                result = await self.analyze_company(symbol)
                if result:
                    results[symbol] = result
                    
            if not results:
                return {"error": "No analysis results available"}
                
            # Generate comparison metrics
            comparison = self._generate_comparison_analysis(results)
            
            return {
                "individual_results": results,
                "comparison": comparison,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in company comparison: {e}")
            return {"error": str(e)}
    def _generate_comparison_analysis(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparative analysis of multiple companies"""
        comparison = {
            "rankings": {},
            "statistics": {},
            "insights": []
        }
        
        try:
            # Extract scores for ranking
            spr_scores = {symbol: result['spr_score'] for symbol, result in results.items()}
            profit_scores = {symbol: result['profit_performance'] for symbol, result in results.items()}
            sustainability_scores = {symbol: result['sustainability_impact'] for symbol, result in results.items()}
            
            # Generate rankings
            comparison["rankings"] = {
                "spr_score": sorted(spr_scores.items(), key=lambda x: x[1], reverse=True),
                "profit_performance": sorted(profit_scores.items(), key=lambda x: x[1], reverse=True),
                "sustainability_impact": sorted(sustainability_scores.items(), key=lambda x: x[1], reverse=True)
            }
            
            # Calculate statistics
            comparison["statistics"] = {
                "average_spr": np.mean(list(spr_scores.values())),
                "spr_std": np.std(list(spr_scores.values())),
                "best_performer": max(spr_scores.items(), key=lambda x: x[1]),
                "most_sustainable": max(sustainability_scores.items(), key=lambda x: x[1]),
                "most_profitable": max(profit_scores.items(), key=lambda x: x[1])
            }
            
            # Generate insights
            best_company = comparison["statistics"]["best_performer"][0]
            comparison["insights"].append(
                f"{best_company} has the highest SPR score of {comparison['statistics']['best_performer'][1]:.2f}"
            )
            
        except Exception as e:
            self.logger.warning(f"Error generating comparison analysis: {e}")
            
        return comparison
        
    def get_historical_spr_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Retrieve historical SPR data for a company
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with historical data
        """
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM spr_results 
                WHERE symbol = ? AND created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days), (symbol,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {"message": f"No historical data found for {symbol}"}
                
            # Convert to structured format
            historical_data = []
            for row in rows:
                historical_data.append({
                    "date": row[9],  # created_at
                    "spr_score": row[3],
                    "profit_performance_score": row[4],
                    "sustainability_impact_score": row[5],
                    "research_alignment_score": row[6],
                    "confidence_level": row[8]
                })
                
            return {
                "symbol": symbol,
                "data": historical_data,
                "count": len(historical_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {e}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize analyzer
        analyzer = SPRAnalyzer()
        
        # Test with a single company
        print("=== Single Company Analysis ===")
        result = await analyzer.analyze_company("TSLA")
        if result:
            print(f"Company: {result.company_name}")
            print(f"SPR Score: {result.spr_score:.2f}/10")
            print(f"Profit Performance: {result.profit_performance_score:.2f}/10")
            print(f"Sustainability Impact: {result.sustainability_impact_score:.2f}/10")
            print(f"Research Alignment: {result.research_alignment_score:.2f}/10")
            print(f"Confidence Level: {result.confidence_level:.2f}")
            print("\nKey Findings:")
            for finding in result.key_findings:
                print(f"• {finding}")
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"• {rec}")
                
        # Test comparison
        print("\n=== Company Comparison ===")
        comparison = await analyzer.compare_companies(["TSLA", "GOOGL"])
        if "comparison" in comparison:
            rankings = comparison["comparison"]["rankings"]["spr_score"]
            print("SPR Score Rankings:")
            for i, (symbol, score) in enumerate(rankings, 1):
                print(f"{i}. {symbol}: {score:.2f}")
    
    # Run the example
    asyncio.run(main())