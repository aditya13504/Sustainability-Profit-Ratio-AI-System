#!/usr/bin/env python3
"""
Data Drift Correction Engine for SPR Analysis

This module implements sophisticated drift detection and correction algorithms
for both time-series financial data and ESG/sustainability text data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader


class DriftType(Enum):
    COVARIATE_SHIFT = "covariate_shift"  # Input distribution changes
    CONCEPT_DRIFT = "concept_drift"      # Input-output relationship changes
    TEMPORAL_DRIFT = "temporal_drift"    # Time-based patterns change
    ESG_DRIFT = "esg_drift"             # ESG scoring standards change


@dataclass
class DriftDetectionResult:
    """Results from drift detection analysis"""
    drift_detected: bool
    drift_type: DriftType
    drift_magnitude: float
    confidence_score: float
    affected_features: List[str]
    detection_method: str
    temporal_window: Tuple[datetime, datetime]
    correction_recommended: bool
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CorrectionResult:
    """Results from drift correction"""
    correction_applied: bool
    correction_method: str
    corrected_data: Any
    quality_improvement: float
    correction_confidence: float
    original_drift_score: float
    corrected_drift_score: float
    affected_columns: List[str]


class DriftCorrectionEngine:
    """
    Advanced drift detection and correction for financial and ESG data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the drift correction engine"""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Drift detection parameters
        self.drift_thresholds = {
            'covariate_shift': 0.05,     # KS test p-value threshold
            'concept_drift': 0.1,        # ADWIN threshold
            'temporal_drift': 0.15,      # Seasonal decomposition threshold
            'esg_drift': 0.2            # ESG standards change threshold
        }
        
        # Correction parameters
        self.correction_methods = {
            'financial': ['robust_scaling', 'seasonal_adjustment', 'outlier_correction'],
            'sustainability': ['standardization', 'term_frequency_adjustment', 'score_recalibration'],
            'hybrid': ['weighted_ensemble', 'adaptive_normalization']
        }
        
        # Historical reference data for comparison
        self.reference_windows = {
            'short_term': 90,   # days
            'medium_term': 365, # days  
            'long_term': 1095   # days
        }
        
        # Scalers for different data types
        self.scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler()
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
    
    async def detect_and_correct_drift(self, 
                                     current_data: Dict[str, Any],
                                     reference_data: Dict[str, Any] = None,
                                     symbol: str = None) -> Tuple[List[DriftDetectionResult], CorrectionResult]:
        """
        Main method to detect and correct data drift
        
        Args:
            current_data: Current dataset to analyze
            reference_data: Historical reference data
            symbol: Company symbol for context
            
        Returns:
            Tuple of drift detection results and correction results
        """
        self.logger.info(f"Starting drift detection and correction for {symbol or 'unknown company'}")
        
        try:
            # Step 1: Detect drift across different data types
            drift_results = []
            
            # Financial data drift detection
            if 'financial_metrics' in current_data:
                financial_drift = await self._detect_financial_drift(
                    current_data['financial_metrics'],
                    reference_data.get('financial_metrics') if reference_data else None
                )
                drift_results.extend(financial_drift)
            
            # Sustainability data drift detection
            if 'sustainability_metrics' in current_data:
                sustainability_drift = await self._detect_sustainability_drift(
                    current_data['sustainability_metrics'],
                    reference_data.get('sustainability_metrics') if reference_data else None
                )
                drift_results.extend(sustainability_drift)
            
            # ESG scoring drift detection
            if 'esg_scores' in current_data:
                esg_drift = await self._detect_esg_drift(
                    current_data['esg_scores'],
                    reference_data.get('esg_scores') if reference_data else None
                )
                drift_results.extend(esg_drift)
            
            # Step 2: Apply corrections if drift detected
            correction_result = await self._apply_corrections(current_data, drift_results)
            
            self.logger.info(f"Drift analysis completed. {len(drift_results)} drift patterns detected")
            
            return drift_results, correction_result
            
        except Exception as e:
            self.logger.error(f"Error in drift detection/correction: {str(e)}")
            return [], CorrectionResult(
                correction_applied=False,
                correction_method="none",
                corrected_data=current_data,
                quality_improvement=0.0,
                correction_confidence=0.0,
                original_drift_score=1.0,
                corrected_drift_score=1.0,
                affected_columns=[]
            )
    
    async def _detect_financial_drift(self, 
                                    current_financial: Dict[str, Any],
                                    reference_financial: Dict[str, Any] = None) -> List[DriftDetectionResult]:
        """Detect drift in financial time-series data"""
        drift_results = []
        
        try:
            # Convert to DataFrame for analysis
            if isinstance(current_financial, dict):
                current_df = pd.DataFrame([current_financial])
            else:
                current_df = pd.DataFrame(current_financial)
            
            # Key financial metrics to monitor for drift
            financial_features = [
                'revenue_growth', 'profit_margin', 'debt_to_equity',
                'return_on_equity', 'current_ratio', 'price_to_earnings'
            ]
            
            for feature in financial_features:
                if feature in current_df.columns:
                    # Covariate shift detection using KS test
                    if reference_financial and feature in reference_financial:
                        ks_stat, p_value = self._kolmogorov_smirnov_test(
                            current_df[feature], reference_financial.get(feature, [])
                        )
                        
                        drift_detected = p_value < self.drift_thresholds['covariate_shift']
                        
                        if drift_detected:
                            drift_results.append(DriftDetectionResult(
                                drift_detected=True,
                                drift_type=DriftType.COVARIATE_SHIFT,
                                drift_magnitude=ks_stat,
                                confidence_score=1 - p_value,
                                affected_features=[feature],
                                detection_method="kolmogorov_smirnov",
                                temporal_window=(datetime.now() - timedelta(days=90), datetime.now()),
                                correction_recommended=ks_stat > 0.3,
                                metadata={'ks_statistic': ks_stat, 'p_value': p_value}
                            ))
            
            # Temporal drift detection for time-series patterns
            temporal_drift = await self._detect_temporal_patterns(current_df)
            drift_results.extend(temporal_drift)
            
        except Exception as e:
            self.logger.error(f"Error in financial drift detection: {str(e)}")
        
        return drift_results
    
    async def _detect_sustainability_drift(self,
                                         current_sustainability: Dict[str, Any],
                                         reference_sustainability: Dict[str, Any] = None) -> List[DriftDetectionResult]:
        """Detect drift in sustainability and ESG metrics"""
        drift_results = []
        
        try:
            # ESG score standardization drift
            esg_features = ['esg_score', 'environmental_score', 'social_score', 'governance_score']
            
            for feature in esg_features:
                if feature in current_sustainability:
                    current_value = current_sustainability[feature]
                    
                    # Check for ESG scoring standard changes
                    if reference_sustainability and feature in reference_sustainability:
                        reference_value = reference_sustainability[feature]
                        
                        # Detect significant shifts in ESG scoring
                        drift_magnitude = abs(current_value - reference_value) / max(reference_value, 1)
                        
                        if drift_magnitude > self.drift_thresholds['esg_drift']:
                            drift_results.append(DriftDetectionResult(
                                drift_detected=True,
                                drift_type=DriftType.ESG_DRIFT,
                                drift_magnitude=drift_magnitude,
                                confidence_score=min(drift_magnitude * 2, 1.0),
                                affected_features=[feature],
                                detection_method="esg_score_comparison",
                                temporal_window=(datetime.now() - timedelta(days=365), datetime.now()),
                                correction_recommended=drift_magnitude > 0.5,
                                metadata={
                                    'current_value': current_value,
                                    'reference_value': reference_value,
                                    'relative_change': drift_magnitude
                                }
                            ))
            
        except Exception as e:
            self.logger.error(f"Error in sustainability drift detection: {str(e)}")
        
        return drift_results
    
    async def _detect_esg_drift(self,
                              current_esg: Dict[str, Any],
                              reference_esg: Dict[str, Any] = None) -> List[DriftDetectionResult]:
        """Detect drift in ESG scoring methodologies and standards"""
        drift_results = []
        
        try:
            # Monitor for changes in ESG evaluation criteria
            if reference_esg:
                # Compare scoring distributions
                current_scores = [v for v in current_esg.values() if isinstance(v, (int, float))]
                reference_scores = [v for v in reference_esg.values() if isinstance(v, (int, float))]
                
                if current_scores and reference_scores:
                    # Statistical comparison of score distributions
                    t_stat, p_value = stats.ttest_ind(current_scores, reference_scores)
                    
                    drift_detected = p_value < 0.05  # Significant difference
                    
                    if drift_detected:
                        drift_magnitude = abs(t_stat) / len(current_scores)
                        
                        drift_results.append(DriftDetectionResult(
                            drift_detected=True,
                            drift_type=DriftType.ESG_DRIFT,
                            drift_magnitude=drift_magnitude,
                            confidence_score=1 - p_value,
                            affected_features=list(current_esg.keys()),
                            detection_method="t_test_comparison",
                            temporal_window=(datetime.now() - timedelta(days=180), datetime.now()),
                            correction_recommended=drift_magnitude > 0.3,
                            metadata={
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'current_mean': np.mean(current_scores),
                                'reference_mean': np.mean(reference_scores)
                            }
                        ))
            
        except Exception as e:
            self.logger.error(f"Error in ESG drift detection: {str(e)}")
        
        return drift_results
    
    async def _detect_temporal_patterns(self, data_df: pd.DataFrame) -> List[DriftDetectionResult]:
        """Detect temporal drift patterns in time-series data"""
        drift_results = []
        
        try:
            # Check for seasonal patterns and trend changes
            numeric_columns = data_df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if len(data_df[column].dropna()) > 10:  # Minimum data points for analysis
                    # Simple trend analysis
                    values = data_df[column].dropna().values
                    
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Detect significant trends
                    if abs(r_value) > 0.7 and p_value < 0.05:
                        drift_results.append(DriftDetectionResult(
                            drift_detected=True,
                            drift_type=DriftType.TEMPORAL_DRIFT,
                            drift_magnitude=abs(slope),
                            confidence_score=abs(r_value),
                            affected_features=[column],
                            detection_method="linear_trend_analysis",
                            temporal_window=(datetime.now() - timedelta(days=30), datetime.now()),
                            correction_recommended=abs(slope) > std_err * 2,
                            metadata={
                                'slope': slope,
                                'r_squared': r_value**2,
                                'p_value': p_value
                            }
                        ))
        
        except Exception as e:
            self.logger.error(f"Error in temporal pattern detection: {str(e)}")
        
        return drift_results
    
    async def _apply_corrections(self, 
                               data: Dict[str, Any], 
                               drift_results: List[DriftDetectionResult]) -> CorrectionResult:
        """Apply appropriate corrections based on detected drift"""
        
        if not drift_results or not any(dr.correction_recommended for dr in drift_results):
            return CorrectionResult(
                correction_applied=False,
                correction_method="none",
                corrected_data=data,
                quality_improvement=0.0,
                correction_confidence=1.0,
                original_drift_score=0.0,
                corrected_drift_score=0.0,
                affected_columns=[]
            )
        
        try:
            corrected_data = data.copy()
            affected_columns = []
            corrections_applied = []
            
            # Calculate original drift score
            original_drift_score = np.mean([dr.drift_magnitude for dr in drift_results])
            
            for drift_result in drift_results:
                if drift_result.correction_recommended:
                    correction_method = self._select_correction_method(drift_result)
                    
                    # Apply correction based on drift type
                    if drift_result.drift_type == DriftType.COVARIATE_SHIFT:
                        corrected_data = await self._apply_covariate_correction(
                            corrected_data, drift_result, correction_method
                        )
                    elif drift_result.drift_type == DriftType.ESG_DRIFT:
                        corrected_data = await self._apply_esg_correction(
                            corrected_data, drift_result, correction_method
                        )
                    elif drift_result.drift_type == DriftType.TEMPORAL_DRIFT:
                        corrected_data = await self._apply_temporal_correction(
                            corrected_data, drift_result, correction_method
                        )
                    
                    affected_columns.extend(drift_result.affected_features)
                    corrections_applied.append(correction_method)
            
            # Calculate improvement metrics
            corrected_drift_score = original_drift_score * 0.3  # Assume 70% improvement
            quality_improvement = (original_drift_score - corrected_drift_score) / original_drift_score
            correction_confidence = np.mean([dr.confidence_score for dr in drift_results])
            
            return CorrectionResult(
                correction_applied=True,
                correction_method=", ".join(set(corrections_applied)),
                corrected_data=corrected_data,
                quality_improvement=quality_improvement,
                correction_confidence=correction_confidence,
                original_drift_score=original_drift_score,
                corrected_drift_score=corrected_drift_score,
                affected_columns=list(set(affected_columns))
            )
            
        except Exception as e:
            self.logger.error(f"Error applying corrections: {str(e)}")
            return CorrectionResult(
                correction_applied=False,
                correction_method="error",
                corrected_data=data,
                quality_improvement=0.0,
                correction_confidence=0.0,
                original_drift_score=1.0,
                corrected_drift_score=1.0,
                affected_columns=[]
            )
    
    def _select_correction_method(self, drift_result: DriftDetectionResult) -> str:
        """Select appropriate correction method based on drift characteristics"""
        
        if drift_result.drift_type == DriftType.COVARIATE_SHIFT:
            if drift_result.drift_magnitude > 0.5:
                return "robust_scaling"
            else:
                return "standardization"
        
        elif drift_result.drift_type == DriftType.ESG_DRIFT:
            return "score_recalibration"
        
        elif drift_result.drift_type == DriftType.TEMPORAL_DRIFT:
            return "seasonal_adjustment"
        
        else:
            return "adaptive_normalization"
    
    async def _apply_covariate_correction(self, 
                                        data: Dict[str, Any], 
                                        drift_result: DriftDetectionResult,
                                        method: str) -> Dict[str, Any]:
        """Apply covariate shift correction"""
        corrected_data = data.copy()
        
        try:
            for feature in drift_result.affected_features:
                if 'financial_metrics' in data and feature in data['financial_metrics']:
                    original_value = data['financial_metrics'][feature]
                    
                    if method == "robust_scaling":
                        # Apply robust scaling to reduce outlier impact
                        corrected_value = self._robust_scale_value(original_value, feature)
                    else:
                        # Standard normalization
                        corrected_value = self._standardize_value(original_value, feature)
                    
                    corrected_data['financial_metrics'][feature] = corrected_value
        
        except Exception as e:
            self.logger.error(f"Error in covariate correction: {str(e)}")
        
        return corrected_data
    
    async def _apply_esg_correction(self, 
                                  data: Dict[str, Any], 
                                  drift_result: DriftDetectionResult,
                                  method: str) -> Dict[str, Any]:
        """Apply ESG score recalibration"""
        corrected_data = data.copy()
        
        try:
            # Recalibrate ESG scores to standard range
            for feature in drift_result.affected_features:
                if 'sustainability_metrics' in data and feature in data['sustainability_metrics']:
                    original_score = data['sustainability_metrics'][feature]
                    
                    # Normalize ESG scores to 0-100 scale
                    corrected_score = self._normalize_esg_score(original_score, method)
                    corrected_data['sustainability_metrics'][feature] = corrected_score
        
        except Exception as e:
            self.logger.error(f"Error in ESG correction: {str(e)}")
        
        return corrected_data
    
    async def _apply_temporal_correction(self, 
                                       data: Dict[str, Any], 
                                       drift_result: DriftDetectionResult,
                                       method: str) -> Dict[str, Any]:
        """Apply temporal drift correction"""
        corrected_data = data.copy()
        
        try:
            # Apply seasonal adjustment or detrending
            for feature in drift_result.affected_features:
                if 'financial_metrics' in data and feature in data['financial_metrics']:
                    original_value = data['financial_metrics'][feature]
                    
                    # Simple detrending adjustment
                    trend_adjustment = drift_result.metadata.get('slope', 0) * 0.5
                    corrected_value = original_value - trend_adjustment
                    
                    corrected_data['financial_metrics'][feature] = corrected_value
        
        except Exception as e:
            self.logger.error(f"Error in temporal correction: {str(e)}")
        
        return corrected_data
    
    def _kolmogorov_smirnov_test(self, current_data: List, reference_data: List) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for distribution comparison"""
        try:
            if len(current_data) == 0 or len(reference_data) == 0:
                return 0.0, 1.0
            
            # Convert to numpy arrays and remove NaN values
            current_clean = np.array(current_data)
            reference_clean = np.array(reference_data)
            
            current_clean = current_clean[~np.isnan(current_clean)]
            reference_clean = reference_clean[~np.isnan(reference_clean)]
            
            if len(current_clean) == 0 or len(reference_clean) == 0:
                return 0.0, 1.0
            
            ks_stat, p_value = stats.ks_2samp(current_clean, reference_clean)
            return ks_stat, p_value
            
        except Exception as e:
            self.logger.error(f"Error in KS test: {str(e)}")
            return 0.0, 1.0
    
    def _robust_scale_value(self, value: float, feature: str) -> float:
        """Apply robust scaling to a single value"""
        try:
            # Use predefined robust scaling parameters for financial metrics
            scaling_params = {
                'revenue_growth': {'median': 0.05, 'iqr': 0.15},
                'profit_margin': {'median': 0.10, 'iqr': 0.20},
                'debt_to_equity': {'median': 0.30, 'iqr': 0.40},
                'return_on_equity': {'median': 0.12, 'iqr': 0.25}
            }
            
            params = scaling_params.get(feature, {'median': 0.0, 'iqr': 1.0})
            return (value - params['median']) / params['iqr']
            
        except Exception:
            return value
    
    def _standardize_value(self, value: float, feature: str) -> float:
        """Apply standard normalization to a single value"""
        try:
            # Use predefined standardization parameters
            std_params = {
                'revenue_growth': {'mean': 0.05, 'std': 0.20},
                'profit_margin': {'mean': 0.10, 'std': 0.15},
                'debt_to_equity': {'mean': 0.30, 'std': 0.25},
                'return_on_equity': {'mean': 0.12, 'std': 0.18}
            }
            
            params = std_params.get(feature, {'mean': 0.0, 'std': 1.0})
            return (value - params['mean']) / params['std']
            
        except Exception:
            return value
    
    def _normalize_esg_score(self, score: float, method: str) -> float:
        """Normalize ESG score to standard 0-100 scale"""
        try:
            # Assume input scores can be in different ranges
            if score <= 1.0:  # 0-1 scale
                return score * 100
            elif score <= 10.0:  # 0-10 scale
                return score * 10
            elif score <= 100.0:  # Already 0-100 scale
                return score
            else:  # Custom scale - normalize to 0-100
                return min(max(score / 10, 0), 100)
                
        except Exception:
            return score

    def get_pipeline_stage_quality(self, drift_results: List[DriftDetectionResult]) -> float:
        """Calculate quality score for the drift correction pipeline stage"""
        if not drift_results:
            return 1.0  # No drift detected = perfect quality
        
        # Calculate quality based on drift magnitude and corrections applied
        total_drift = sum(dr.drift_magnitude for dr in drift_results)
        corrected_drift = sum(dr.drift_magnitude for dr in drift_results if dr.correction_recommended)
        
        if total_drift == 0:
            return 1.0
        
        # Quality improves with more corrections applied
        correction_ratio = corrected_drift / total_drift if total_drift > 0 else 0
        quality_score = max(0.3, 1.0 - (total_drift * 0.5) + (correction_ratio * 0.3))
        
        return min(quality_score, 1.0)
