"""
Industry-Specific Metrics Module for SPR Analyzer

This module provides tailored sustainability and financial metrics
for different industry sectors to enhance SPR calculation accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logging_utils import get_logger


class IndustryType(Enum):
    """Enumeration of industry types"""
    TECHNOLOGY = "technology"
    AUTOMOTIVE = "automotive"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    CONSUMER_GOODS = "consumer_goods"
    MANUFACTURING = "manufacturing"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"
    AEROSPACE = "aerospace"
    AGRICULTURE = "agriculture"
    RETAIL = "retail"
    MEDIA = "media"
    TRANSPORTATION = "transportation"


@dataclass
class IndustryMetrics:
    """Industry-specific metrics configuration"""
    industry: IndustryType
    sustainability_metrics: Dict[str, Dict[str, Any]]
    financial_metrics: Dict[str, Dict[str, Any]]
    benchmarks: Dict[str, float]
    weights: Dict[str, float]
    sector_risks: List[str]
    opportunities: List[str]


@dataclass
class IndustryAnalysis:
    """Results from industry-specific analysis"""
    industry: IndustryType
    sustainability_score: float
    financial_score: float
    benchmark_comparison: Dict[str, float]
    risk_assessment: Dict[str, float]
    improvement_areas: List[str]
    competitive_position: str


class IndustrySpecificAnalyzer:
    """
    Provides industry-specific analysis for SPR calculations
    """
    
    def __init__(self, config_path: str = None):
        """Initialize industry-specific analyzer"""
        self.config = ConfigLoader(config_path).config
        self.logger = get_logger(__name__)
        
        # Initialize industry metrics
        self.industry_metrics = self._load_industry_metrics()
        
        # Load industry benchmarks
        self.benchmarks = self._load_industry_benchmarks()
        
    def _load_industry_metrics(self) -> Dict[IndustryType, IndustryMetrics]:
        """Load industry-specific metrics configurations"""
        
        metrics = {}
        
        # Technology Industry
        metrics[IndustryType.TECHNOLOGY] = IndustryMetrics(
            industry=IndustryType.TECHNOLOGY,
            sustainability_metrics={
                "energy_efficiency": {
                    "weight": 0.25,
                    "description": "Data center and office energy efficiency",
                    "unit": "PUE (Power Usage Effectiveness)",
                    "target_range": (1.1, 1.3)
                },
                "carbon_footprint": {
                    "weight": 0.20,
                    "description": "Carbon emissions per revenue",
                    "unit": "tCO2e per $M revenue",
                    "target_range": (0, 50)
                },
                "circular_economy": {
                    "weight": 0.15,
                    "description": "Product lifecycle and recycling",
                    "unit": "% recycled materials",
                    "target_range": (70, 100)
                },
                "digital_inclusion": {
                    "weight": 0.15,
                    "description": "Digital divide reduction efforts",
                    "unit": "accessibility score",
                    "target_range": (70, 100)
                },
                "supply_chain_ethics": {
                    "weight": 0.15,
                    "description": "Ethical sourcing practices",
                    "unit": "supplier compliance %",
                    "target_range": (90, 100)
                },
                "innovation_sustainability": {
                    "weight": 0.10,
                    "description": "R&D investment in sustainable tech",
                    "unit": "% of R&D budget",
                    "target_range": (20, 50)
                }
            },
            financial_metrics={
                "revenue_growth": {"weight": 0.25, "benchmark": 0.15},
                "r_and_d_intensity": {"weight": 0.20, "benchmark": 0.12},
                "gross_margin": {"weight": 0.20, "benchmark": 0.65},
                "operating_margin": {"weight": 0.15, "benchmark": 0.25},
                "cash_conversion_cycle": {"weight": 0.10, "benchmark": 30},
                "debt_to_equity": {"weight": 0.10, "benchmark": 0.3}
            },
            benchmarks={
                "sustainability_score": 75.0,
                "carbon_intensity": 25.0,
                "energy_efficiency": 1.2,
                "innovation_investment": 0.15
            },
            weights={
                "sustainability": 0.4,
                "financial": 0.6
            },
            sector_risks=[
                "Rapid technological obsolescence",
                "Cybersecurity threats",
                "Regulatory changes (data privacy)",
                "Supply chain disruptions",
                "Talent retention challenges"
            ],
            opportunities=[
                "AI and automation efficiency gains",
                "Cloud computing growth",
                "Sustainable technology development",
                "Digital transformation services",
                "IoT and smart city solutions"
            ]
        )
        
        # Automotive Industry
        metrics[IndustryType.AUTOMOTIVE] = IndustryMetrics(
            industry=IndustryType.AUTOMOTIVE,
            sustainability_metrics={
                "ev_transition": {
                    "weight": 0.30,
                    "description": "Electric vehicle production ratio",
                    "unit": "% of total production",
                    "target_range": (30, 100)
                },
                "manufacturing_emissions": {
                    "weight": 0.20,
                    "description": "Manufacturing carbon intensity",
                    "unit": "tCO2e per vehicle",
                    "target_range": (0, 5)
                },
                "material_sustainability": {
                    "weight": 0.15,
                    "description": "Sustainable materials usage",
                    "unit": "% recycled/bio materials",
                    "target_range": (40, 80)
                },
                "water_usage": {
                    "weight": 0.10,
                    "description": "Water consumption efficiency",
                    "unit": "L per vehicle produced",
                    "target_range": (1000, 3000)
                },
                "end_of_life_management": {
                    "weight": 0.15,
                    "description": "Vehicle recycling programs",
                    "unit": "% recyclability",
                    "target_range": (85, 95)
                },
                "supplier_sustainability": {
                    "weight": 0.10,
                    "description": "Supplier ESG compliance",
                    "unit": "% compliant suppliers",
                    "target_range": (80, 100)
                }
            },
            financial_metrics={
                "operating_margin": {"weight": 0.25, "benchmark": 0.08},
                "inventory_turnover": {"weight": 0.20, "benchmark": 12},
                "capex_intensity": {"weight": 0.20, "benchmark": 0.06},
                "working_capital_ratio": {"weight": 0.15, "benchmark": -0.10},
                "debt_to_equity": {"weight": 0.10, "benchmark": 0.8},
                "ev_investment_ratio": {"weight": 0.10, "benchmark": 0.25}
            },
            benchmarks={
                "sustainability_score": 60.0,
                "ev_percentage": 25.0,
                "carbon_intensity": 8.0,
                "recyclability": 90.0
            },
            weights={
                "sustainability": 0.5,
                "financial": 0.5
            },
            sector_risks=[
                "EV transition costs",
                "Battery supply chain risks",
                "Regulatory emissions standards",
                "Changing consumer preferences",
                "Autonomous vehicle disruption"
            ],
            opportunities=[
                "Electric vehicle market growth",
                "Autonomous vehicle development",
                "Mobility-as-a-Service",
                "Battery technology advancement",
                "Sustainable transportation solutions"
            ]
        )
        
        # Energy Industry
        metrics[IndustryType.ENERGY] = IndustryMetrics(
            industry=IndustryType.ENERGY,
            sustainability_metrics={
                "renewable_portfolio": {
                    "weight": 0.35,
                    "description": "Renewable energy generation ratio",
                    "unit": "% of total capacity",
                    "target_range": (50, 100)
                },
                "carbon_intensity": {
                    "weight": 0.25,
                    "description": "Carbon emissions per MWh",
                    "unit": "tCO2e per MWh",
                    "target_range": (0, 200)
                },
                "methane_emissions": {
                    "weight": 0.15,
                    "description": "Methane leak reduction",
                    "unit": "% reduction from baseline",
                    "target_range": (50, 90)
                },
                "water_consumption": {
                    "weight": 0.10,
                    "description": "Water usage efficiency",
                    "unit": "L per MWh",
                    "target_range": (0, 1000)
                },
                "biodiversity_impact": {
                    "weight": 0.10,
                    "description": "Land use and biodiversity protection",
                    "unit": "biodiversity impact score",
                    "target_range": (70, 100)
                },
                "grid_modernization": {
                    "weight": 0.05,
                    "description": "Smart grid investments",
                    "unit": "% of capex in grid tech",
                    "target_range": (15, 30)
                }
            },
            financial_metrics={
                "ebitda_margin": {"weight": 0.30, "benchmark": 0.35},
                "capex_to_revenue": {"weight": 0.20, "benchmark": 0.15},
                "debt_to_ebitda": {"weight": 0.20, "benchmark": 3.5},
                "roe": {"weight": 0.15, "benchmark": 0.12},
                "dividend_yield": {"weight": 0.10, "benchmark": 0.04},
                "renewable_capex_ratio": {"weight": 0.05, "benchmark": 0.60}
            },
            benchmarks={
                "sustainability_score": 55.0,
                "renewable_percentage": 40.0,
                "carbon_intensity": 300.0,
                "methane_reduction": 60.0
            },
            weights={
                "sustainability": 0.6,
                "financial": 0.4
            },
            sector_risks=[
                "Stranded fossil fuel assets",
                "Regulatory carbon pricing",
                "Renewable energy intermittency",
                "Grid stability challenges",
                "Energy transition costs"
            ],
            opportunities=[
                "Renewable energy expansion",
                "Energy storage development",
                "Carbon capture technologies",
                "Hydrogen economy",
                "Grid modernization services"
            ]
        )
        
        # Add more industries as needed...
        
        self.logger.info(f"Loaded metrics for {len(metrics)} industries")
        return metrics
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry benchmark data"""
        
        benchmarks = {
            "technology": {
                "avg_sustainability_score": 75.0,
                "top_quartile_sustainability": 85.0,
                "avg_carbon_intensity": 25.0,
                "avg_energy_efficiency": 1.2,
                "avg_revenue_growth": 0.15,
                "avg_operating_margin": 0.25
            },
            "automotive": {
                "avg_sustainability_score": 60.0,
                "top_quartile_sustainability": 75.0,
                "avg_ev_percentage": 25.0,
                "avg_carbon_intensity": 8.0,
                "avg_operating_margin": 0.08,
                "avg_capex_intensity": 0.06
            },
            "energy": {
                "avg_sustainability_score": 55.0,
                "top_quartile_sustainability": 70.0,
                "avg_renewable_percentage": 40.0,
                "avg_carbon_intensity": 300.0,
                "avg_ebitda_margin": 0.35,
                "avg_debt_to_ebitda": 3.5
            }
        }
        
        return benchmarks
    
    def identify_industry(self, company_data: Dict[str, Any]) -> IndustryType:
        """
        Identify the industry type for a company
        
        Args:
            company_data: Company information including sector, description, etc.
            
        Returns:
            IndustryType enum
        """
        
        # Industry keywords mapping
        industry_keywords = {
            IndustryType.TECHNOLOGY: [
                "software", "technology", "internet", "cloud", "artificial intelligence",
                "cybersecurity", "semiconductor", "electronics", "computing"
            ],
            IndustryType.AUTOMOTIVE: [
                "automotive", "automobile", "vehicle", "transportation equipment",
                "auto", "motor", "electric vehicle", "ev"
            ],
            IndustryType.ENERGY: [
                "energy", "oil", "gas", "petroleum", "renewable", "solar", "wind",
                "utilities", "power", "electricity", "nuclear"
            ],
            IndustryType.HEALTHCARE: [
                "healthcare", "pharmaceutical", "biotechnology", "medical",
                "health", "biotech", "drug", "medicine"
            ],
            IndustryType.FINANCIAL_SERVICES: [
                "financial", "banking", "insurance", "investment", "finance",
                "bank", "capital", "credit", "mortgage"
            ],
            IndustryType.CONSUMER_GOODS: [
                "consumer", "retail", "food", "beverage", "apparel", "textile",
                "household", "personal care", "cosmetics"
            ],
            IndustryType.MANUFACTURING: [
                "manufacturing", "industrial", "machinery", "equipment",
                "construction", "materials", "chemicals"
            ]
        }
        
        # Get company description/sector information
        company_info = str(company_data.get('sector', '')) + " " + \
                      str(company_data.get('industry', '')) + " " + \
                      str(company_data.get('description', ''))
        company_info = company_info.lower()
        
        # Score each industry based on keyword matches
        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in company_info)
            if score > 0:
                industry_scores[industry] = score
        
        # Return industry with highest score, default to TECHNOLOGY
        if industry_scores:
            return max(industry_scores.keys(), key=lambda x: industry_scores[x])
        else:
            return IndustryType.TECHNOLOGY
    
    def calculate_industry_specific_score(self, company_data: Dict[str, Any], 
                                        industry: IndustryType) -> IndustryAnalysis:
        """
        Calculate industry-specific SPR score
        
        Args:
            company_data: Company sustainability and financial data
            industry: Industry type
            
        Returns:
            IndustryAnalysis object
        """
        
        if industry not in self.industry_metrics:
            self.logger.warning(f"No metrics defined for {industry}, using default")
            industry = IndustryType.TECHNOLOGY
        
        metrics = self.industry_metrics[industry]
        
        # Calculate sustainability score
        sustainability_score = self._calculate_sustainability_score(
            company_data, metrics.sustainability_metrics
        )
        
        # Calculate financial score
        financial_score = self._calculate_financial_score(
            company_data, metrics.financial_metrics
        )
        
        # Benchmark comparison
        benchmark_comparison = self._compare_to_benchmarks(
            company_data, industry, sustainability_score, financial_score
        )
        
        # Risk assessment
        risk_assessment = self._assess_industry_risks(
            company_data, metrics.sector_risks
        )
        
        # Identify improvement areas
        improvement_areas = self._identify_improvement_areas(
            company_data, metrics.sustainability_metrics, metrics.financial_metrics
        )
        
        # Determine competitive position
        competitive_position = self._determine_competitive_position(
            sustainability_score, financial_score, benchmark_comparison
        )
        
        return IndustryAnalysis(
            industry=industry,
            sustainability_score=sustainability_score,
            financial_score=financial_score,
            benchmark_comparison=benchmark_comparison,
            risk_assessment=risk_assessment,
            improvement_areas=improvement_areas,
            competitive_position=competitive_position
        )
    
    def _calculate_sustainability_score(self, company_data: Dict[str, Any], 
                                      sustainability_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted sustainability score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric_config in sustainability_metrics.items():
            weight = metric_config["weight"]
            target_range = metric_config["target_range"]
            
            # Get metric value from company data
            metric_value = company_data.get(metric_name, 0)
            
            # Normalize to 0-100 scale
            min_val, max_val = target_range
            if max_val > min_val:
                normalized_score = max(0, min(100, 
                    100 * (metric_value - min_val) / (max_val - min_val)
                ))
            else:
                normalized_score = 50  # Default if no range
            
            total_score += normalized_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_financial_score(self, company_data: Dict[str, Any], 
                                 financial_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted financial score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric_config in financial_metrics.items():
            weight = metric_config["weight"]
            benchmark = metric_config["benchmark"]
            
            # Get metric value from company data
            metric_value = company_data.get(metric_name, 0)
            
            # Score based on performance vs benchmark
            if benchmark != 0:
                ratio = metric_value / benchmark
                # Cap the score to prevent extreme values
                score = max(0, min(150, 100 * ratio))
            else:
                score = 50  # Default if no benchmark
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _compare_to_benchmarks(self, company_data: Dict[str, Any], 
                             industry: IndustryType, 
                             sustainability_score: float, 
                             financial_score: float) -> Dict[str, float]:
        """Compare company performance to industry benchmarks"""
        
        industry_name = industry.value
        benchmarks = self.benchmarks.get(industry_name, {})
        
        comparison = {}
        
        # Sustainability comparison
        avg_sustainability = benchmarks.get("avg_sustainability_score", 60.0)
        top_quartile = benchmarks.get("top_quartile_sustainability", 80.0)
        
        comparison["sustainability_vs_average"] = sustainability_score / avg_sustainability
        comparison["sustainability_vs_top_quartile"] = sustainability_score / top_quartile
        
        # Financial comparison
        comparison["financial_vs_average"] = financial_score / 100.0  # Assuming 100 is average
        
        # Specific metric comparisons
        for metric, benchmark_value in benchmarks.items():
            if metric.startswith("avg_") and metric.replace("avg_", "") in company_data:
                metric_name = metric.replace("avg_", "")
                company_value = company_data[metric_name]
                if benchmark_value != 0:
                    comparison[f"{metric_name}_vs_benchmark"] = company_value / benchmark_value
        
        return comparison
    
    def _assess_industry_risks(self, company_data: Dict[str, Any], 
                             sector_risks: List[str]) -> Dict[str, float]:
        """Assess company exposure to industry-specific risks"""
        
        risk_assessment = {}
        
        # This is a simplified risk assessment
        # In practice, this would involve more sophisticated analysis
        
        for risk in sector_risks:
            # Basic risk scoring based on company characteristics
            risk_score = 0.5  # Default medium risk
            
            # Adjust based on company data
            if "transition" in risk.lower() and "renewable" in company_data.get("business_model", "").lower():
                risk_score = 0.3  # Lower risk if already transitioning
            elif "regulatory" in risk.lower() and company_data.get("compliance_score", 50) > 80:
                risk_score = 0.2  # Lower risk with good compliance
            elif "supply chain" in risk.lower() and company_data.get("supplier_diversity", 0) > 0.7:
                risk_score = 0.3  # Lower risk with diverse suppliers
            
            risk_assessment[risk] = risk_score
        
        return risk_assessment
    
    def _identify_improvement_areas(self, company_data: Dict[str, Any], 
                                  sustainability_metrics: Dict[str, Dict[str, Any]], 
                                  financial_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify areas where the company can improve"""
        
        improvement_areas = []
        
        # Check sustainability metrics
        for metric_name, metric_config in sustainability_metrics.items():
            metric_value = company_data.get(metric_name, 0)
            target_range = metric_config["target_range"]
            min_val, max_val = target_range
            
            # If performance is below 70% of target range
            target_70_percent = min_val + 0.7 * (max_val - min_val)
            if metric_value < target_70_percent:
                improvement_areas.append(f"Improve {metric_name.replace('_', ' ')}")
        
        # Check financial metrics
        for metric_name, metric_config in financial_metrics.items():
            metric_value = company_data.get(metric_name, 0)
            benchmark = metric_config["benchmark"]
            
            # If performance is below 80% of benchmark
            if benchmark != 0 and metric_value < 0.8 * benchmark:
                improvement_areas.append(f"Enhance {metric_name.replace('_', ' ')}")
        
        return improvement_areas[:5]  # Return top 5 areas
    
    def _determine_competitive_position(self, sustainability_score: float, 
                                      financial_score: float, 
                                      benchmark_comparison: Dict[str, float]) -> str:
        """Determine company's competitive position"""
        
        # Combined score
        combined_score = (sustainability_score + financial_score) / 2
        
        # Benchmark performance
        sustainability_vs_avg = benchmark_comparison.get("sustainability_vs_average", 1.0)
        financial_vs_avg = benchmark_comparison.get("financial_vs_average", 1.0)
        
        # Determine position
        if combined_score >= 80 and sustainability_vs_avg >= 1.2 and financial_vs_avg >= 1.1:
            return "Market Leader"
        elif combined_score >= 70 and sustainability_vs_avg >= 1.0 and financial_vs_avg >= 1.0:
            return "Strong Performer"
        elif combined_score >= 60:
            return "Average Performer"
        elif combined_score >= 40:
            return "Below Average"
        else:
            return "Laggard"
    
    def get_industry_recommendations(self, industry: IndustryType, 
                                   analysis: IndustryAnalysis) -> List[str]:
        """Get industry-specific recommendations for improvement"""
        
        recommendations = []
        
        if industry not in self.industry_metrics:
            return recommendations
        
        metrics = self.industry_metrics[industry]
        
        # Add specific recommendations based on industry
        if industry == IndustryType.TECHNOLOGY:
            if analysis.sustainability_score < 70:
                recommendations.extend([
                    "Invest in renewable energy for data centers",
                    "Implement circular design principles for products",
                    "Enhance supply chain transparency and ethics",
                    "Develop AI solutions for sustainability challenges"
                ])
        
        elif industry == IndustryType.AUTOMOTIVE:
            if analysis.sustainability_score < 60:
                recommendations.extend([
                    "Accelerate electric vehicle production",
                    "Implement sustainable manufacturing processes",
                    "Develop vehicle recycling programs",
                    "Partner with renewable energy providers"
                ])
        
        elif industry == IndustryType.ENERGY:
            if analysis.sustainability_score < 55:
                recommendations.extend([
                    "Increase renewable energy portfolio",
                    "Implement carbon capture technologies",
                    "Reduce methane emissions",
                    "Invest in energy storage solutions"
                ])
        
        # Add financial recommendations
        if analysis.financial_score < 70:
            recommendations.extend([
                "Optimize operational efficiency",
                "Diversify revenue streams",
                "Improve working capital management",
                "Enhance digital transformation initiatives"
            ])
        
        return recommendations[:8]  # Return top 8 recommendations


# Example usage
if __name__ == "__main__":
    # Example company data
    sample_company_data = {
        "sector": "Technology",
        "industry": "Software",
        "energy_efficiency": 1.15,
        "carbon_footprint": 30,
        "circular_economy": 65,
        "digital_inclusion": 80,
        "supply_chain_ethics": 85,
        "innovation_sustainability": 25,
        "revenue_growth": 0.18,
        "r_and_d_intensity": 0.15,
        "gross_margin": 0.72,
        "operating_margin": 0.28,
        "cash_conversion_cycle": 25,
        "debt_to_equity": 0.2
    }
    
    # Initialize analyzer
    analyzer = IndustrySpecificAnalyzer()
    
    # Identify industry
    industry = analyzer.identify_industry(sample_company_data)
    print(f"Identified industry: {industry.value}")
    
    # Perform industry-specific analysis
    analysis = analyzer.calculate_industry_specific_score(sample_company_data, industry)
    
    print(f"\nIndustry Analysis Results:")
    print(f"Sustainability Score: {analysis.sustainability_score:.1f}")
    print(f"Financial Score: {analysis.financial_score:.1f}")
    print(f"Competitive Position: {analysis.competitive_position}")
    print(f"Improvement Areas: {', '.join(analysis.improvement_areas[:3])}")
    
    # Get recommendations
    recommendations = analyzer.get_industry_recommendations(industry, analysis)
    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")
