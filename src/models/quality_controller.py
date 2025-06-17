#!/usr/bin/env python3
"""
Advanced Data Quality Controller for SPR Analysis

This module implements comprehensive quality control mechanisms including
data completeness, accuracy validation, consistency checks, and outlier detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for data assessment"""
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    outlier_score: float
    overall_quality: float
    
    # Detailed breakdowns
    missing_fields: List[str]
    invalid_values: List[str]
    outliers_detected: List[str]
    consistency_issues: List[str]
    quality_flags: Dict[str, Any]


class DataQualityController:
    """
    Advanced quality control system for financial and sustainability data
    with automated filtering and correction capabilities.
    """
    
    def __init__(self, config: Dict):
        """Initialize the quality controller"""
        self.config = config
        self.logger = self._setup_logging()
        
        # Quality thresholds
        self.quality_thresholds = {
            'financial_completeness': 0.85,
            'sustainability_completeness': 0.75,
            'data_freshness_days': 90,
            'outlier_threshold': 3.0,
            'consistency_threshold': 0.8,
            'overall_minimum': 0.7
        }
        
        # Valid ranges for financial metrics
        self.financial_ranges = {
            'revenue': (0, float('inf')),
            'profit_margin': (-1.0, 1.0),
            'total_assets': (0, float('inf')),
            'debt_to_equity': (0, 20.0),
            'current_ratio': (0, 10.0),
            'return_on_equity': (-1.0, 2.0),
            'return_on_assets': (-1.0, 1.0),
            'gross_margin': (-1.0, 1.0),
            'operating_margin': (-1.0, 1.0),
            'ebitda_margin': (-2.0, 1.0)
        }
        
        # Valid ranges for sustainability metrics
        self.sustainability_ranges = {
            'esg_score': (0, 100),
            'environmental_score': (0, 100),
            'social_score': (0, 100),
            'governance_score': (0, 100),
            'carbon_footprint': (0, float('inf')),
            'renewable_energy_percentage': (0, 100),
            'waste_reduction_percentage': (0, 100),
            'employee_satisfaction': (0, 100),
            'board_diversity_percentage': (0, 100)
        }
        
        # Critical fields that must be present
        self.required_financial_fields = [
            'revenue', 'total_assets', 'profit_margin', 'debt_to_equity'
        ]
        
        self.required_sustainability_fields = [
            'esg_score', 'environmental_score', 'social_score', 'governance_score'
        ]
    
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
    
    def assess_financial_data_quality(self, financial_data: Dict) -> QualityMetrics:
        """
        Comprehensive assessment of financial data quality
        
        Args:
            financial_data: Dictionary containing financial metrics
            
        Returns:
            QualityMetrics object with detailed quality assessment
        """
        try:
            # 1. Completeness Assessment
            completeness_result = self._assess_completeness(
                financial_data, self.required_financial_fields
            )
            
            # 2. Accuracy Assessment
            accuracy_result = self._assess_financial_accuracy(financial_data)
            
            # 3. Consistency Assessment
            consistency_result = self._assess_financial_consistency(financial_data)
            
            # 4. Timeliness Assessment
            timeliness_result = self._assess_data_timeliness(financial_data)
            
            # 5. Outlier Detection
            outlier_result = self._detect_financial_outliers(financial_data)
            
            # Calculate overall quality
            overall_quality = np.mean([
                completeness_result['score'],
                accuracy_result['score'],
                consistency_result['score'],
                timeliness_result['score'],
                outlier_result['score']
            ])
            
            return QualityMetrics(
                completeness_score=completeness_result['score'],
                accuracy_score=accuracy_result['score'],
                consistency_score=consistency_result['score'],
                timeliness_score=timeliness_result['score'],
                outlier_score=outlier_result['score'],
                overall_quality=overall_quality,
                missing_fields=completeness_result['missing'],
                invalid_values=accuracy_result['invalid'],
                outliers_detected=outlier_result['outliers'],
                consistency_issues=consistency_result['issues'],
                quality_flags={
                    'completeness_passed': completeness_result['score'] >= self.quality_thresholds['financial_completeness'],
                    'accuracy_passed': accuracy_result['score'] >= 0.8,
                    'consistency_passed': consistency_result['score'] >= self.quality_thresholds['consistency_threshold'],
                    'timeliness_passed': timeliness_result['score'] >= 0.7,
                    'outliers_acceptable': outlier_result['score'] >= 0.8
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing financial data quality: {str(e)}")
            return self._create_default_quality_metrics()
    
    def assess_sustainability_data_quality(self, sustainability_data: Dict) -> QualityMetrics:
        """
        Comprehensive assessment of sustainability/ESG data quality
        
        Args:
            sustainability_data: Dictionary containing sustainability metrics
            
        Returns:
            QualityMetrics object with detailed quality assessment
        """
        try:
            # 1. Completeness Assessment
            completeness_result = self._assess_completeness(
                sustainability_data, self.required_sustainability_fields
            )
            
            # 2. Accuracy Assessment
            accuracy_result = self._assess_sustainability_accuracy(sustainability_data)
            
            # 3. Consistency Assessment
            consistency_result = self._assess_sustainability_consistency(sustainability_data)
            
            # 4. Timeliness Assessment
            timeliness_result = self._assess_data_timeliness(sustainability_data)
            
            # 5. Outlier Detection
            outlier_result = self._detect_sustainability_outliers(sustainability_data)
            
            # Calculate overall quality
            overall_quality = np.mean([
                completeness_result['score'],
                accuracy_result['score'],
                consistency_result['score'],
                timeliness_result['score'],
                outlier_result['score']
            ])
            
            return QualityMetrics(
                completeness_score=completeness_result['score'],
                accuracy_score=accuracy_result['score'],
                consistency_score=consistency_result['score'],
                timeliness_score=timeliness_result['score'],
                outlier_score=outlier_result['score'],
                overall_quality=overall_quality,
                missing_fields=completeness_result['missing'],
                invalid_values=accuracy_result['invalid'],
                outliers_detected=outlier_result['outliers'],
                consistency_issues=consistency_result['issues'],
                quality_flags={
                    'completeness_passed': completeness_result['score'] >= self.quality_thresholds['sustainability_completeness'],
                    'accuracy_passed': accuracy_result['score'] >= 0.8,
                    'consistency_passed': consistency_result['score'] >= self.quality_thresholds['consistency_threshold'],
                    'timeliness_passed': timeliness_result['score'] >= 0.7,
                    'outliers_acceptable': outlier_result['score'] >= 0.8
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing sustainability data quality: {str(e)}")
            return self._create_default_quality_metrics()
    
    def filter_low_quality_inputs(self, data: Dict, min_quality: float = 0.7) -> Tuple[Dict, bool]:
        """
        Filter out low-quality data inputs and clean the dataset
        
        Args:
            data: Complete dataset to filter
            min_quality: Minimum quality threshold
            
        Returns:
            Tuple of (filtered_data, quality_passed)
        """
        try:
            filtered_data = data.copy()
            quality_issues = []
            
            # Assess financial data quality
            if 'financial_metrics' in data:
                financial_quality = self.assess_financial_data_quality(data['financial_metrics'])
                
                if financial_quality.overall_quality < min_quality:
                    # Remove problematic fields
                    filtered_data['financial_metrics'] = self._clean_financial_data(
                        data['financial_metrics'], financial_quality
                    )
                    quality_issues.append('financial_data_cleaned')
                
                # Flag if still below threshold
                if financial_quality.overall_quality < min_quality * 0.8:
                    quality_issues.append('financial_data_poor_quality')
            
            # Assess sustainability data quality
            if 'sustainability_metrics' in data:
                sustainability_quality = self.assess_sustainability_data_quality(data['sustainability_metrics'])
                
                if sustainability_quality.overall_quality < min_quality:
                    # Remove problematic fields
                    filtered_data['sustainability_metrics'] = self._clean_sustainability_data(
                        data['sustainability_metrics'], sustainability_quality
                    )
                    quality_issues.append('sustainability_data_cleaned')
                
                # Flag if still below threshold
                if sustainability_quality.overall_quality < min_quality * 0.8:
                    quality_issues.append('sustainability_data_poor_quality')
            
            # Overall quality assessment
            overall_passed = len([issue for issue in quality_issues if 'poor_quality' in issue]) == 0
            
            # Add quality metadata
            filtered_data['quality_metadata'] = {
                'quality_issues': quality_issues,
                'quality_passed': overall_passed,
                'filtering_applied': len(quality_issues) > 0,
                'min_quality_threshold': min_quality
            }
            
            return filtered_data, overall_passed
            
        except Exception as e:
            self.logger.error(f"Error filtering low-quality inputs: {str(e)}")
            return data, False
    
    def _assess_completeness(self, data: Dict, required_fields: List[str]) -> Dict[str, Any]:
        """Assess data completeness"""
        missing_fields = []
        present_fields = 0
        
        for field in required_fields:
            if field in data and data[field] is not None:
                present_fields += 1
            else:
                missing_fields.append(field)
        
        completeness_score = present_fields / len(required_fields) if required_fields else 1.0
        
        return {
            'score': completeness_score,
            'missing': missing_fields,
            'present': present_fields,
            'total_required': len(required_fields)
        }
    
    def _assess_financial_accuracy(self, financial_data: Dict) -> Dict[str, Any]:
        """Assess accuracy of financial data using business logic"""
        invalid_values = []
        total_checks = 0
        passed_checks = 0
        
        for field, value in financial_data.items():
            if field in self.financial_ranges and value is not None:
                total_checks += 1
                min_val, max_val = self.financial_ranges[field]
                
                if min_val <= value <= max_val:
                    passed_checks += 1
                else:
                    invalid_values.append(f"{field}: {value} outside range [{min_val}, {max_val}]")
        
        # Additional business logic checks
        if 'revenue' in financial_data and 'total_assets' in financial_data:
            total_checks += 1
            if financial_data['revenue'] > 0 and financial_data['total_assets'] > 0:
                asset_turnover = financial_data['revenue'] / financial_data['total_assets']
                if 0.1 <= asset_turnover <= 10:  # Reasonable asset turnover
                    passed_checks += 1
                else:
                    invalid_values.append(f"Asset turnover ratio suspicious: {asset_turnover}")
        
        # Check for impossible combinations
        if 'gross_margin' in financial_data and 'operating_margin' in financial_data:
            total_checks += 1
            if financial_data['gross_margin'] >= financial_data['operating_margin']:
                passed_checks += 1
            else:
                invalid_values.append("Operating margin higher than gross margin")
        
        accuracy_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'score': accuracy_score,
            'invalid': invalid_values,
            'checks_passed': passed_checks,
            'total_checks': total_checks
        }
    
    def _assess_sustainability_accuracy(self, sustainability_data: Dict) -> Dict[str, Any]:
        """Assess accuracy of sustainability data"""
        invalid_values = []
        total_checks = 0
        passed_checks = 0
        
        for field, value in sustainability_data.items():
            if field in self.sustainability_ranges and value is not None:
                total_checks += 1
                min_val, max_val = self.sustainability_ranges[field]
                
                if min_val <= value <= max_val:
                    passed_checks += 1
                else:
                    invalid_values.append(f"{field}: {value} outside range [{min_val}, {max_val}]")
        
        # Check ESG component consistency
        esg_components = ['environmental_score', 'social_score', 'governance_score']
        if all(comp in sustainability_data for comp in esg_components) and 'esg_score' in sustainability_data:
            total_checks += 1
            avg_components = np.mean([sustainability_data[comp] for comp in esg_components])
            esg_score = sustainability_data['esg_score']
            
            # ESG score should be reasonably close to component average
            if abs(esg_score - avg_components) <= 15:  # Within 15 points
                passed_checks += 1
            else:
                invalid_values.append(f"ESG score inconsistent with components: {esg_score} vs {avg_components}")
        
        accuracy_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'score': accuracy_score,
            'invalid': invalid_values,
            'checks_passed': passed_checks,
            'total_checks': total_checks
        }
    
    def _assess_financial_consistency(self, financial_data: Dict) -> Dict[str, Any]:
        """Assess internal consistency of financial data"""
        consistency_issues = []
        total_checks = 0
        passed_checks = 0
        
        # Check ratio consistency
        if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
            total_checks += 1
            if financial_data['current_liabilities'] > 0:
                calculated_ratio = financial_data['current_assets'] / financial_data['current_liabilities']
                reported_ratio = financial_data.get('current_ratio')
                
                if reported_ratio and abs(calculated_ratio - reported_ratio) / reported_ratio <= 0.05:
                    passed_checks += 1
                elif not reported_ratio:
                    passed_checks += 1  # No reported ratio to compare
                else:
                    consistency_issues.append("Current ratio calculation inconsistent")
        
        # Check profitability consistency
        if 'net_income' in financial_data and 'revenue' in financial_data:
            total_checks += 1
            if financial_data['revenue'] > 0:
                calculated_margin = financial_data['net_income'] / financial_data['revenue']
                reported_margin = financial_data.get('profit_margin')
                
                if reported_margin and abs(calculated_margin - reported_margin) <= 0.02:
                    passed_checks += 1
                elif not reported_margin:
                    passed_checks += 1
                else:
                    consistency_issues.append("Profit margin calculation inconsistent")
        
        consistency_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'score': consistency_score,
            'issues': consistency_issues,
            'checks_passed': passed_checks,
            'total_checks': total_checks
        }
    
    def _assess_sustainability_consistency(self, sustainability_data: Dict) -> Dict[str, Any]:
        """Assess internal consistency of sustainability data"""
        consistency_issues = []
        total_checks = 0
        passed_checks = 0
        
        # Check ESG score consistency with components
        esg_components = ['environmental_score', 'social_score', 'governance_score']
        if all(comp in sustainability_data for comp in esg_components):
            total_checks += 1
            component_scores = [sustainability_data[comp] for comp in esg_components]
            
            # Check for extreme variations (one component much different from others)
            if max(component_scores) - min(component_scores) <= 40:  # Within 40 points
                passed_checks += 1
            else:
                consistency_issues.append("Large variation in ESG component scores")
        
        # Check environmental metrics consistency
        env_metrics = ['carbon_footprint', 'renewable_energy_percentage', 'environmental_score']
        if all(metric in sustainability_data for metric in env_metrics):
            total_checks += 1
            # High renewable energy should correlate with better environmental score
            renewable_pct = sustainability_data['renewable_energy_percentage']
            env_score = sustainability_data['environmental_score']
            
            # Expect positive correlation
            if (renewable_pct >= 50 and env_score >= 60) or (renewable_pct < 50 and env_score < 80):
                passed_checks += 1
            else:
                consistency_issues.append("Renewable energy % inconsistent with environmental score")
        
        consistency_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'score': consistency_score,
            'issues': consistency_issues,
            'checks_passed': passed_checks,
            'total_checks': total_checks
        }
    
    def _assess_data_timeliness(self, data: Dict) -> Dict[str, Any]:
        """Assess timeliness of data"""
        timeliness_score = 1.0  # Default to perfect if no timestamp
        
        if 'timestamp' in data or 'last_updated' in data or 'report_date' in data:
            timestamp_field = next((field for field in ['timestamp', 'last_updated', 'report_date'] if field in data), None)
            
            if timestamp_field:
                try:
                    if isinstance(data[timestamp_field], str):
                        # Try to parse string timestamps
                        data_date = pd.to_datetime(data[timestamp_field])
                    else:
                        data_date = pd.to_datetime(data[timestamp_field])
                    
                    current_date = pd.Timestamp.now()
                    days_old = (current_date - data_date).days
                    
                    # Calculate timeliness score based on age
                    if days_old <= self.quality_thresholds['data_freshness_days']:
                        timeliness_score = 1.0
                    elif days_old <= self.quality_thresholds['data_freshness_days'] * 2:
                        timeliness_score = 0.8
                    elif days_old <= self.quality_thresholds['data_freshness_days'] * 4:
                        timeliness_score = 0.6
                    else:
                        timeliness_score = 0.4
                        
                except Exception:
                    timeliness_score = 0.7  # Unknown format, moderate score
        
        return {
            'score': timeliness_score,
            'issues': [] if timeliness_score >= 0.8 else ['Data may be stale']
        }
    
    def _detect_financial_outliers(self, financial_data: Dict) -> Dict[str, Any]:
        """Detect outliers in financial data using statistical methods"""
        outliers = []
        numeric_values = []
        
        # Collect numeric values for outlier detection
        for field, value in financial_data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_values.append((field, value))
        
        if len(numeric_values) < 3:
            return {'score': 1.0, 'outliers': []}
        
        # Use IQR method for outlier detection within each metric type
        ratio_fields = [item for item in numeric_values if 'ratio' in item[0] or 'margin' in item[0]]
        
        if len(ratio_fields) >= 3:
            ratio_values = [item[1] for item in ratio_fields]
            q1, q3 = np.percentile(ratio_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for field, value in ratio_fields:
                if value < lower_bound or value > upper_bound:
                    outliers.append(f"{field}: {value} (outside [{lower_bound:.3f}, {upper_bound:.3f}])")
        
        # Calculate outlier score
        outlier_score = max(0, 1.0 - len(outliers) * 0.2)
        
        return {
            'score': outlier_score,
            'outliers': outliers,
            'outlier_count': len(outliers)
        }
    
    def _detect_sustainability_outliers(self, sustainability_data: Dict) -> Dict[str, Any]:
        """Detect outliers in sustainability data"""
        outliers = []
        
        # Check for extreme values that might indicate data errors
        extreme_checks = [
            ('esg_score', 0, 100),
            ('environmental_score', 0, 100),
            ('social_score', 0, 100),
            ('governance_score', 0, 100),
            ('renewable_energy_percentage', 0, 100),
            ('waste_reduction_percentage', 0, 100)
        ]
        
        for field, min_val, max_val in extreme_checks:
            if field in sustainability_data:
                value = sustainability_data[field]
                if value < min_val or value > max_val:
                    outliers.append(f"{field}: {value} outside expected range [{min_val}, {max_val}]")
        
        # Check for suspicious perfect scores (might indicate missing data filled with defaults)
        perfect_score_fields = ['esg_score', 'environmental_score', 'social_score', 'governance_score']
        perfect_scores = sum(1 for field in perfect_score_fields 
                           if field in sustainability_data and sustainability_data[field] == 100)
        
        if perfect_scores >= 3:
            outliers.append("Suspicious: Multiple perfect scores (possible data quality issue)")
        
        outlier_score = max(0, 1.0 - len(outliers) * 0.15)
        
        return {
            'score': outlier_score,
            'outliers': outliers,
            'outlier_count': len(outliers)
        }
    
    def _clean_financial_data(self, financial_data: Dict, quality_metrics: QualityMetrics) -> Dict:
        """Clean financial data by removing/correcting problematic fields"""
        cleaned_data = financial_data.copy()
        
        # Remove fields with invalid values
        for invalid_value in quality_metrics.invalid_values:
            field_name = invalid_value.split(':')[0]
            if field_name in cleaned_data:
                self.logger.warning(f"Removing invalid financial field: {field_name}")
                # Don't remove, but flag for special handling
                cleaned_data[f"{field_name}_quality_flag"] = "invalid_value_detected"
        
        # Handle outliers
        for outlier in quality_metrics.outliers_detected:
            field_name = outlier.split(':')[0]
            if field_name in cleaned_data:
                # Flag outliers rather than removing them
                cleaned_data[f"{field_name}_quality_flag"] = "outlier_detected"
        
        return cleaned_data
    
    def _clean_sustainability_data(self, sustainability_data: Dict, quality_metrics: QualityMetrics) -> Dict:
        """Clean sustainability data by removing/correcting problematic fields"""
        cleaned_data = sustainability_data.copy()
        
        # Handle invalid values
        for invalid_value in quality_metrics.invalid_values:
            field_name = invalid_value.split(':')[0]
            if field_name in cleaned_data:
                self.logger.warning(f"Flagging invalid sustainability field: {field_name}")
                cleaned_data[f"{field_name}_quality_flag"] = "invalid_value_detected"
        
        # Handle outliers
        for outlier in quality_metrics.outliers_detected:
            field_name = outlier.split(':')[0]
            if field_name in cleaned_data:
                cleaned_data[f"{field_name}_quality_flag"] = "outlier_detected"
        
        return cleaned_data
    
    def _create_default_quality_metrics(self) -> QualityMetrics:
        """Create default quality metrics for error cases"""
        return QualityMetrics(
            completeness_score=0.5,
            accuracy_score=0.5,
            consistency_score=0.5,
            timeliness_score=0.5,
            outlier_score=0.5,
            overall_quality=0.5,
            missing_fields=[],
            invalid_values=[],
            outliers_detected=[],
            consistency_issues=[],
            quality_flags={'error': 'Quality assessment failed'}
        )
    
    def generate_quality_report(self, financial_quality: QualityMetrics, 
                              sustainability_quality: QualityMetrics) -> Dict[str, Any]:
        """Generate a comprehensive quality report"""
        return {
            'overall_assessment': {
                'financial_quality': financial_quality.overall_quality,
                'sustainability_quality': sustainability_quality.overall_quality,
                'combined_quality': (financial_quality.overall_quality + sustainability_quality.overall_quality) / 2
            },
            'detailed_scores': {
                'financial': {
                    'completeness': financial_quality.completeness_score,
                    'accuracy': financial_quality.accuracy_score,
                    'consistency': financial_quality.consistency_score,
                    'timeliness': financial_quality.timeliness_score,
                    'outlier_detection': financial_quality.outlier_score
                },
                'sustainability': {
                    'completeness': sustainability_quality.completeness_score,
                    'accuracy': sustainability_quality.accuracy_score,
                    'consistency': sustainability_quality.consistency_score,
                    'timeliness': sustainability_quality.timeliness_score,
                    'outlier_detection': sustainability_quality.outlier_score
                }
            },
            'quality_issues': {
                'financial': {
                    'missing_fields': financial_quality.missing_fields,
                    'invalid_values': financial_quality.invalid_values,
                    'outliers': financial_quality.outliers_detected,
                    'consistency_issues': financial_quality.consistency_issues
                },
                'sustainability': {
                    'missing_fields': sustainability_quality.missing_fields,
                    'invalid_values': sustainability_quality.invalid_values,
                    'outliers': sustainability_quality.outliers_detected,
                    'consistency_issues': sustainability_quality.consistency_issues
                }
            },
            'recommendations': self._generate_quality_recommendations(financial_quality, sustainability_quality)
        }
    
    def _generate_quality_recommendations(self, financial_quality: QualityMetrics, 
                                        sustainability_quality: QualityMetrics) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Financial data recommendations
        if financial_quality.completeness_score < 0.8:
            recommendations.append("Improve financial data completeness by collecting missing metrics")
        
        if financial_quality.accuracy_score < 0.8:
            recommendations.append("Review financial data accuracy - some values appear invalid")
        
        if len(financial_quality.outliers_detected) > 2:
            recommendations.append("Investigate financial outliers that may indicate data errors")
        
        # Sustainability data recommendations
        if sustainability_quality.completeness_score < 0.7:
            recommendations.append("Enhance sustainability data collection for better ESG coverage")
        
        if sustainability_quality.consistency_score < 0.8:
            recommendations.append("Address sustainability data consistency issues")
        
        # Overall recommendations
        overall_quality = (financial_quality.overall_quality + sustainability_quality.overall_quality) / 2
        if overall_quality < 0.7:
            recommendations.append("Consider implementing automated data validation processes")
        
        return recommendations
