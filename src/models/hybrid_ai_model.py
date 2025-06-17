#!/usr/bin/env python3
"""
Hybrid AI Model combining CNNs and Transformers for SPR Analysis

This module implements a sophisticated hybrid architecture that processes
financial time-series data with CNNs and sustainability text data with Transformers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, pipeline
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader


@dataclass
class ModelOutput:
    """Output structure for hybrid model predictions"""
    spr_score: float
    profit_score: float
    sustainability_score: float
    research_score: float
    risk_factor: float
    confidence_score: float
    feature_importance: Dict[str, float]
    explanation: Dict[str, Any]
    model_version: str


class FinancialCNN(nn.Module):
    """CNN for processing financial time-series data"""
    
    def __init__(self, input_channels: int, sequence_length: int, output_dim: int = 128):
        super(FinancialCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        
        # 1D Convolution layers for time-series processing
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling and dropout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        
        # Convolutional layers with activation and normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        pooled = self.pool(x).squeeze(-1)  # (batch_size, 256)
        
        # Attention mechanism for interpretability
        x_transposed = x.transpose(1, 2)  # (batch_size, sequence_length, 256)
        attended, attention_weights = self.attention(x_transposed, x_transposed, x_transposed)
        attended_pooled = attended.mean(dim=1)  # (batch_size, 256)
        
        # Combine pooled and attended features
        combined = pooled + attended_pooled
        
        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, attention_weights


class SustainabilityTransformer(nn.Module):
    """Transformer for processing sustainability text and ESG data"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", output_dim: int = 128):
        super(SustainabilityTransformer, self).__init__()
        
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
          # Freeze transformer weights initially
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        # Handle different model architectures
        if hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'layer'):
            # BERT-style models
            for param in self.transformer.encoder.layer[-2:].parameters():
                param.requires_grad = True
        elif hasattr(self.transformer, 'transformer') and hasattr(self.transformer.transformer, 'layer'):
            # DistilBERT-style models
            for param in self.transformer.transformer.layer[-2:].parameters():
                param.requires_grad = True
        else:
            # Fallback: unfreeze all parameters
            for param in self.transformer.parameters():
                param.requires_grad = True
        
        # Classification and regression heads
        hidden_size = self.transformer.config.hidden_size
        
        self.esg_processor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Multi-head attention for text importance
        self.text_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, text_inputs, esg_metrics=None):
        # Process text inputs
        transformer_outputs = self.transformer(**text_inputs)
        
        # Get pooled representation
        last_hidden_state = transformer_outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Apply attention for interpretability
        attended_output, text_attention_weights = self.text_attention(
            last_hidden_state, last_hidden_state, last_hidden_state
        )
        attended_pooled = attended_output.mean(dim=1)
        
        # Process text features
        text_features = self.text_classifier(attended_pooled)
        
        # Process ESG metrics if provided
        if esg_metrics is not None:
            esg_features = self.esg_processor(esg_metrics)
            # Combine text and ESG features
            combined_features = text_features + esg_features
        else:
            combined_features = text_features
        
        return combined_features, text_attention_weights


class HybridSPRModel(nn.Module):
    """
    Hybrid model combining CNN for financial data and Transformer for sustainability text
    """
    
    def __init__(self, config: Dict):
        super(HybridSPRModel, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.financial_input_dim = config.get('financial_input_dim', 10)
        self.sequence_length = config.get('sequence_length', 252)  # 1 year of trading days
        self.transformer_model = config.get('transformer_model', 'distilbert-base-uncased')
        
        # Initialize submodels
        self.financial_cnn = FinancialCNN(
            input_channels=self.financial_input_dim,
            sequence_length=self.sequence_length,
            output_dim=128
        )
        
        self.sustainability_transformer = SustainabilityTransformer(
            model_name=self.transformer_model,
            output_dim=128
        )
        
        # Fusion and prediction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 256),  # 128 + 128 = 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-task prediction heads
        self.spr_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # SPR score between 0 and 1 (scaled to 0-10 later)
        )
        
        self.profit_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.sustainability_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.research_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.risk_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        self.logger = self._setup_logging()
        
        # Data scalers
        self.financial_scaler = StandardScaler()
        self.esg_scaler = MinMaxScaler()
        
        # Model metadata
        self.model_version = "1.0.0"
        self.is_trained = False
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
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
    
    def forward(self, financial_data, sustainability_text, esg_metrics=None):
        """Forward pass through the hybrid model"""
        
        # Process financial data with CNN
        financial_features, financial_attention = self.financial_cnn(financial_data)
        
        # Process sustainability data with Transformer
        sustainability_features, text_attention = self.sustainability_transformer(
            sustainability_text, esg_metrics
        )
        
        # Fuse features
        combined_features = torch.cat([financial_features, sustainability_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Multi-task predictions
        spr_score = self.spr_predictor(fused_features)
        profit_score = self.profit_predictor(fused_features)
        sustainability_score = self.sustainability_predictor(fused_features)
        research_score = self.research_predictor(fused_features)
        risk_factor = self.risk_predictor(fused_features)
        confidence = self.confidence_estimator(fused_features)
        
        return {
            'spr_score': spr_score,
            'profit_score': profit_score,
            'sustainability_score': sustainability_score,
            'research_score': research_score,
            'risk_factor': risk_factor,
            'confidence': confidence,
            'fused_features': fused_features,
            'financial_attention': financial_attention,
            'text_attention': text_attention
        }
    
    async def analyze(self, financial_features: Dict, sustainability_features: Dict, symbol: str) -> Dict[str, Any]:
        """
        Perform SPR analysis using the hybrid model
        
        Args:
            financial_features: Financial time-series data
            sustainability_features: Sustainability text and ESG data
            symbol: Company symbol
            
        Returns:
            ModelOutput with predictions and explanations
        """
        try:
            self.eval()
            
            with torch.no_grad():
                # Prepare financial data
                financial_tensor = self._prepare_financial_tensor(financial_features)
                
                # Prepare sustainability text
                text_inputs = self._prepare_text_inputs(sustainability_features)
                
                # Prepare ESG metrics
                esg_tensor = self._prepare_esg_tensor(sustainability_features) if sustainability_features.get('sustainability_metrics') else None
                
                # Run forward pass
                outputs = self.forward(financial_tensor, text_inputs, esg_tensor)
                
                # Convert to scores (0-10 scale)
                spr_score = outputs['spr_score'].item() * 10
                profit_score = outputs['profit_score'].item() * 10
                sustainability_score = outputs['sustainability_score'].item() * 10
                research_score = outputs['research_score'].item() * 10
                risk_factor = outputs['risk_factor'].item()
                confidence_score = outputs['confidence'].item()
                
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(outputs)
                
                # Generate explanation
                explanation = self._generate_explanation(outputs, financial_features, sustainability_features)
                
                return {
                    'spr_score': spr_score,
                    'profit_score': profit_score,
                    'sustainability_score': sustainability_score,
                    'research_score': research_score,
                    'risk_factor': risk_factor,
                    'confidence_score': confidence_score,
                    'feature_importance': feature_importance,
                    'explanation': explanation,
                    'model_version': self.model_version,
                    'insights': self._generate_insights(spr_score, profit_score, sustainability_score, research_score)
                }
                
        except Exception as e:
            self.logger.error(f"Error in hybrid model analysis: {str(e)}")
            # Return fallback predictions
            return self._generate_fallback_predictions()
    
    def _prepare_financial_tensor(self, financial_features: Dict) -> torch.Tensor:
        """Convert financial features to tensor format"""
        try:
            # Extract time-series financial data
            financial_metrics = financial_features.get('financial_metrics', {})
            
            # Create feature matrix
            features = []
            feature_names = ['revenue', 'profit_margin', 'total_assets', 'debt_to_equity', 
                           'current_ratio', 'return_on_equity', 'return_on_assets',
                           'gross_margin', 'operating_margin', 'ebitda_margin']
            
            for feature_name in feature_names:
                # Use current value or default
                value = financial_metrics.get(feature_name, 0.0)
                if isinstance(value, (list, np.ndarray)):
                    # If time-series data available
                    features.append(value[-self.sequence_length:] if len(value) >= self.sequence_length 
                                  else [value[0]] * self.sequence_length)
                else:
                    # If single value, replicate for time series
                    features.append([value] * self.sequence_length)
            
            # Convert to tensor
            feature_matrix = np.array(features)  # (num_features, sequence_length)
            
            # Normalize
            if hasattr(self, 'financial_scaler') and hasattr(self.financial_scaler, 'mean_'):
                feature_matrix = self.financial_scaler.transform(feature_matrix.T).T
            
            # Convert to tensor and add batch dimension
            tensor = torch.FloatTensor(feature_matrix).unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.warning(f"Error preparing financial tensor: {str(e)}")
            # Return default tensor
            default_tensor = torch.zeros(1, self.financial_input_dim, self.sequence_length).to(self.device)
            return default_tensor
    
    def _prepare_text_inputs(self, sustainability_features: Dict) -> Dict[str, torch.Tensor]:
        """Prepare text inputs for transformer"""
        try:
            # Extract sustainability text
            research_insights = sustainability_features.get('research_insights', [])
            
            # Combine research insights into text
            text_content = ""
            for insight in research_insights:
                if isinstance(insight, dict):
                    text_content += f"{insight.get('practice', '')} {insight.get('impact_description', '')} "
                else:
                    text_content += str(insight) + " "
            
            # Add company sustainability information
            sustainability_metrics = sustainability_features.get('sustainability_metrics', {})
            esg_score = sustainability_metrics.get('esg_score', 0)
            text_content += f"ESG score: {esg_score}. "
            
            # Default text if no content
            if not text_content.strip():
                text_content = "Company sustainability analysis corporate environmental social governance"
            
            # Tokenize
            tokenizer = self.sustainability_transformer.tokenizer
            inputs = tokenizer(
                text_content,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            self.logger.warning(f"Error preparing text inputs: {str(e)}")
            # Return default inputs
            tokenizer = self.sustainability_transformer.tokenizer
            default_inputs = tokenizer(
                "sustainability analysis",
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            return {k: v.to(self.device) for k, v in default_inputs.items()}
    
    def _prepare_esg_tensor(self, sustainability_features: Dict) -> torch.Tensor:
        """Prepare ESG metrics tensor"""
        try:
            sustainability_metrics = sustainability_features.get('sustainability_metrics', {})
            
            # Extract ESG metrics
            esg_features = [
                sustainability_metrics.get('esg_score', 50),
                sustainability_metrics.get('environmental_score', 50),
                sustainability_metrics.get('social_score', 50),
                sustainability_metrics.get('governance_score', 50),
                sustainability_metrics.get('carbon_footprint', 0) / 1000,  # Normalize
                sustainability_metrics.get('renewable_energy_percentage', 0) / 100,
                sustainability_metrics.get('waste_reduction_percentage', 0) / 100,
                sustainability_metrics.get('employee_satisfaction', 50) / 100
            ]
            
            # Convert to tensor
            tensor = torch.FloatTensor(esg_features).unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.warning(f"Error preparing ESG tensor: {str(e)}")
            return torch.zeros(1, 8).to(self.device)
    
    def _calculate_feature_importance(self, outputs: Dict) -> Dict[str, float]:
        """Calculate feature importance from attention weights"""
        try:
            feature_importance = {}
            
            # Financial feature importance from CNN attention
            if 'financial_attention' in outputs:
                financial_attn = outputs['financial_attention']
                # Average attention weights across heads and sequence
                financial_importance = financial_attn.mean(dim=(0, 1)).cpu().numpy()
                
                feature_names = ['revenue', 'profit_margin', 'assets', 'debt_equity', 'current_ratio']
                for i, name in enumerate(feature_names[:len(financial_importance)]):
                    feature_importance[f"financial_{name}"] = float(financial_importance[i])
            
            # Text feature importance from transformer attention
            if 'text_attention' in outputs:
                text_attn = outputs['text_attention']
                # Average attention weights
                text_importance = text_attn.mean(dim=(0, 1)).mean().item()
                feature_importance['sustainability_text'] = text_importance
            
            # Normalize importance scores
            if feature_importance:
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def _generate_explanation(self, outputs: Dict, financial_features: Dict, sustainability_features: Dict) -> Dict[str, Any]:
        """Generate explanation for model predictions"""
        try:
            explanation = {
                'model_type': 'Hybrid CNN-Transformer',
                'financial_processing': 'Convolutional Neural Network for time-series analysis',
                'sustainability_processing': 'Transformer for text and ESG data analysis',
                'fusion_method': 'Feature concatenation with multi-layer perceptron',
                'confidence_factors': [],
                'key_drivers': []
            }
            
            # Analyze confidence factors
            confidence = outputs['confidence'].item()
            if confidence > 0.8:
                explanation['confidence_factors'].append('High model confidence due to complete data')
            elif confidence > 0.6:
                explanation['confidence_factors'].append('Moderate confidence with some data limitations')
            else:
                explanation['confidence_factors'].append('Lower confidence due to data quality issues')
            
            # Identify key drivers
            spr_score = outputs['spr_score'].item() * 10
            profit_score = outputs['profit_score'].item() * 10
            sustainability_score = outputs['sustainability_score'].item() * 10
            
            if profit_score > 7:
                explanation['key_drivers'].append('Strong financial performance')
            if sustainability_score > 7:
                explanation['key_drivers'].append('Excellent sustainability practices')
            if spr_score > 8:
                explanation['key_drivers'].append('Outstanding sustainability-profitability alignment')
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Error generating explanation: {str(e)}")
            return {'error': 'Could not generate explanation'}
    
    def _generate_insights(self, spr_score: float, profit_score: float, 
                          sustainability_score: float, research_score: float) -> List[str]:
        """Generate actionable insights from scores"""
        insights = []
        
        if spr_score >= 8:
            insights.append("Excellent sustainability-profitability alignment - maintain current strategies")
        elif spr_score >= 6:
            insights.append("Good SPR performance with room for optimization")
        else:
            insights.append("SPR performance below average - consider strategic sustainability investments")
        
        if profit_score < 5 and sustainability_score > 7:
            insights.append("High sustainability but low profitability - focus on monetizing green initiatives")
        
        if profit_score > 7 and sustainability_score < 5:
            insights.append("Strong profits but weak sustainability - ESG improvements needed for long-term value")
        
        if research_score < 3:
            insights.append("Limited research backing - align practices with evidence-based sustainability strategies")
        
        return insights
    
    def _generate_fallback_predictions(self) -> Dict[str, Any]:
        """Generate fallback predictions when model fails"""
        return {
            'spr_score': 5.0,
            'profit_score': 5.0,
            'sustainability_score': 5.0,
            'research_score': 3.0,
            'risk_factor': 0.5,
            'confidence_score': 0.5,
            'feature_importance': {},
            'explanation': {'error': 'Model analysis failed, using fallback'},
            'model_version': self.model_version,
            'insights': ['Model analysis unavailable - manual review recommended']
        }
    
    def train_model(self, training_data: List[Dict], validation_data: List[Dict], epochs: int = 50):
        """Train the hybrid model"""
        self.logger.info("Starting hybrid model training...")
        
        # Set to training mode
        self.train()
        
        # Initialize optimizer
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(training_data, optimizer)
            val_loss = self._validate_epoch(validation_data)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_best_model()
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.is_trained = True
        self.logger.info("Model training completed")
    
    def _train_epoch(self, training_data: List[Dict], optimizer) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in training_data:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(
                batch['financial_data'],
                batch['text_inputs'],
                batch.get('esg_data')
            )
            
            # Calculate loss
            loss = self._calculate_loss(outputs, batch['targets'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _validate_epoch(self, validation_data: List[Dict]) -> float:
        """Validate for one epoch"""
        total_loss = 0
        num_batches = 0
        
        self.eval()
        with torch.no_grad():
            for batch in validation_data:
                outputs = self.forward(
                    batch['financial_data'],
                    batch['text_inputs'],
                    batch.get('esg_data')
                )
                
                loss = self._calculate_loss(outputs, batch['targets'])
                total_loss += loss.item()
                num_batches += 1
        
        self.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _calculate_loss(self, outputs: Dict, targets: Dict) -> torch.Tensor:
        """Calculate multi-task loss"""
        mse_loss = nn.MSELoss()
        
        # Individual losses
        spr_loss = mse_loss(outputs['spr_score'], targets['spr_score'])
        profit_loss = mse_loss(outputs['profit_score'], targets['profit_score'])
        sustainability_loss = mse_loss(outputs['sustainability_score'], targets['sustainability_score'])
        research_loss = mse_loss(outputs['research_score'], targets['research_score'])
        
        # Weighted combination
        total_loss = (
            0.4 * spr_loss +
            0.25 * profit_loss +
            0.25 * sustainability_loss +
            0.1 * research_loss
        )
        
        return total_loss
    
    def _save_best_model(self):
        """Save the best model checkpoint"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_version': self.model_version,
                'config': self.config
            }, 'models/best_hybrid_spr_model.pth')
        except Exception as e:
            self.logger.warning(f"Could not save model: {str(e)}")
    
    def load_pretrained_model(self, model_path: str):
        """Load a pretrained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.model_version = checkpoint.get('model_version', '1.0.0')
            self.is_trained = True
            self.logger.info(f"Loaded pretrained model version {self.model_version}")
        except Exception as e:
            self.logger.warning(f"Could not load pretrained model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Hybrid CNN-Transformer',
            'version': self.model_version,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
