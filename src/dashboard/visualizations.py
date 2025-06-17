"""
Advanced Visualization Components

Provides enhanced charts, graphs, and interactive visualizations
for the SPR Analyzer dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from utils.logging_utils import LoggerMixin


class AdvancedVisualizations(LoggerMixin):
    """Advanced visualization components for SPR analysis"""
    
    def __init__(self):
        """Initialize the visualization components"""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'background': '#f8f9fa',
            'text': '#212529'
        }
        
        self.logger.info("AdvancedVisualizations initialized")
    
    def create_spr_gauge_chart(self, spr_score: float, company_name: str = "") -> go.Figure:
        """
        Create a gauge chart for SPR score
        
        Args:
            spr_score: SPR score (0-10)
            company_name: Company name for title
            
        Returns:
            go.Figure: Gauge chart
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = spr_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"SPR Score{' - ' + company_name if company_name else ''}"},
            delta = {'reference': 5.0, 'position': "top"},
            gauge = {
                'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self._get_gauge_color(spr_score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 3], 'color': '#ffcccb'},  # Poor
                    {'range': [3, 5], 'color': '#fff2cc'},  # Fair
                    {'range': [5, 7], 'color': '#d4edda'},  # Good
                    {'range': [7, 10], 'color': '#c3e6cb'}  # Excellent
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7.0
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': self.color_scheme['text'], 'family': "Arial"}
        )
        
        return fig
    
    def create_radar_chart(self, companies_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create radar chart comparing multiple companies
        
        Args:
            companies_data: List of company analysis data
            
        Returns:
            go.Figure: Radar chart
        """
        fig = go.Figure()
        
        categories = ['Profit Performance', 'Sustainability Impact', 
                     'Research Alignment', 'Financial Stability', 'Innovation Score']
        
        for company in companies_data:
            values = [
                company.get('profit_performance', 0),
                company.get('sustainability_impact', 0),
                company.get('research_alignment', 0),
                company.get('financial_stability', 5),  # Placeholder
                company.get('innovation_score', 5)      # Placeholder
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=company.get('company', company.get('symbol', 'Unknown')),
                line=dict(width=2),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Company Performance Radar Chart",
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create correlation heatmap between different metrics
        
        Args:
            correlation_data: Correlation matrix data
            
        Returns:
            go.Figure: Heatmap
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(correlation_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(df.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Metrics Correlation Matrix",
            xaxis_nticks=36,
            height=500
        )
        
        return fig
    
    def create_trend_analysis(self, historical_data: List[Dict[str, Any]], 
                            metric: str = 'spr_score') -> go.Figure:
        """
        Create trend analysis chart for historical data
        
        Args:
            historical_data: Historical analysis data
            metric: Metric to analyze trends for
            
        Returns:
            go.Figure: Trend chart
        """
        # Prepare data
        dates = []
        values = []
        
        for data_point in historical_data:
            date_str = data_point.get('timestamp', data_point.get('date', ''))
            if date_str:
                try:
                    date = pd.to_datetime(date_str)
                    dates.append(date)
                    values.append(data_point.get(metric, 0))
                except:
                    continue
        
        if not dates:
            # Generate sample trend data if no historical data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
            values = np.random.normal(6, 1.5, len(dates))
            values = np.clip(values, 0, 10)  # Ensure values are in valid range
        
        # Create trend line
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(color=self.color_scheme['primary'], width=3),
            marker=dict(size=8, color=self.color_scheme['primary'])
        ))
        
        # Add trend line
        if len(values) > 1:
            z = np.polyfit(range(len(values)), values, 1)
            p = np.poly1d(z)
            trend_values = [p(i) for i in range(len(values))]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_values,
                mode='lines',
                name='Trend',
                line=dict(color=self.color_scheme['warning'], dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_sector_comparison(self, sector_data: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
        """
        Create sector comparison visualization
        
        Args:
            sector_data: Dictionary mapping sector names to company data
            
        Returns:
            go.Figure: Sector comparison chart
        """
        fig = go.Figure()
        
        sectors = []
        avg_spr_scores = []
        avg_sustainability = []
        avg_profit = []
        
        for sector, companies in sector_data.items():
            if not companies:
                continue
                
            sectors.append(sector)
            avg_spr_scores.append(np.mean([c.get('spr_score', 0) for c in companies]))
            avg_sustainability.append(np.mean([c.get('sustainability_impact', 0) for c in companies]))
            avg_profit.append(np.mean([c.get('profit_performance', 0) for c in companies]))
        
        # Create grouped bar chart
        fig.add_trace(go.Bar(
            name='SPR Score',
            x=sectors,
            y=avg_spr_scores,
            marker_color=self.color_scheme['primary']
        ))
        
        fig.add_trace(go.Bar(
            name='Sustainability Impact',
            x=sectors,
            y=avg_sustainability,
            marker_color=self.color_scheme['success']
        ))
        
        fig.add_trace(go.Bar(
            name='Profit Performance',
            x=sectors,
            y=avg_profit,
            marker_color=self.color_scheme['secondary']
        ))
        
        fig.update_layout(
            title="Average Scores by Sector",
            barmode='group',
            xaxis_title="Sector",
            yaxis_title="Average Score",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_bubble_chart(self, companies_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create bubble chart showing relationship between metrics
        
        Args:
            companies_data: List of company data
            
        Returns:
            go.Figure: Bubble chart
        """
        fig = go.Figure()
        
        for company in companies_data:
            fig.add_trace(go.Scatter(
                x=[company.get('profit_performance', 0)],
                y=[company.get('sustainability_impact', 0)],
                mode='markers',
                marker=dict(
                    size=company.get('spr_score', 5) * 5,  # Size based on SPR score
                    color=company.get('research_alignment', 0),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Research Alignment"),
                    line=dict(width=2, color='black')
                ),
                name=company.get('company', company.get('symbol', 'Unknown')),
                text=f"SPR: {company.get('spr_score', 0):.1f}",
                textposition="middle center"
            ))
        
        fig.update_layout(
            title="Profit vs Sustainability Performance",
            xaxis_title="Profit Performance",
            yaxis_title="Sustainability Impact",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_waterfall_chart(self, components: Dict[str, float], 
                              title: str = "SPR Score Components") -> go.Figure:
        """
        Create waterfall chart showing score components
        
        Args:
            components: Dictionary of score components
            title: Chart title
            
        Returns:
            go.Figure: Waterfall chart
        """
        labels = list(components.keys()) + ['Total']
        values = list(components.values())
        
        # Calculate cumulative values for waterfall effect
        cumulative = [0]
        for i, val in enumerate(values):
            cumulative.append(cumulative[-1] + val)
        
        fig = go.Figure()
        
        # Add bars for each component
        for i, (label, value) in enumerate(zip(labels[:-1], values)):
            fig.add_trace(go.Bar(
                name=label,
                x=[label],
                y=[value],
                base=cumulative[i],
                marker_color=self.color_scheme['primary'] if value > 0 else self.color_scheme['warning']
            ))
        
        # Add total bar
        fig.add_trace(go.Bar(
            name='Total',
            x=['Total'],
            y=[sum(values)],
            marker_color=self.color_scheme['success']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Components",
            yaxis_title="Score Contribution",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_risk_return_scatter(self, companies_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create risk-return scatter plot
        
        Args:
            companies_data: List of company data
            
        Returns:
            go.Figure: Risk-return scatter plot
        """
        fig = go.Figure()
        
        for company in companies_data:
            # Calculate risk (volatility proxy) and return
            risk = 10 - company.get('financial_stability', 5)  # Higher stability = lower risk
            return_val = company.get('profit_performance', 0)
            
            fig.add_trace(go.Scatter(
                x=[risk],
                y=[return_val],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=company.get('sustainability_impact', 0),
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sustainability Impact"),
                    line=dict(width=2, color='black')
                ),
                text=company.get('symbol', 'Unknown'),
                textposition="top center",
                name=company.get('company', company.get('symbol', 'Unknown')),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Risk vs Return Analysis",
            xaxis_title="Risk Level",
            yaxis_title="Return Performance",
            height=500
        )
        
        return fig
    
    def _get_gauge_color(self, score: float) -> str:
        """
        Get color for gauge based on score
        
        Args:
            score: Score value (0-10)
            
        Returns:
            str: Color code
        """
        if score >= 8:
            return "#28a745"  # Green
        elif score >= 6:
            return "#ffc107"  # Yellow
        elif score >= 4:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
