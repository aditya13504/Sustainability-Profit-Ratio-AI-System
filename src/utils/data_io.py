"""
Data Export/Import Utilities

Handles exporting analysis results and importing historical data
in various formats (CSV, JSON, Excel, PDF).
"""

import json
import csv
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Union, Any
import asyncio

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from utils.logging_utils import LoggerMixin


class DataExporter(LoggerMixin):
    """Handles exporting analysis results to various formats"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the data exporter
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"DataExporter initialized with output directory: {self.output_dir}")
    
    def export_to_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Export data to JSON format
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spr_analysis_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Data exported to JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.log_error(e, "JSON export failed")
            raise
    
    def export_to_csv(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Export data to CSV format
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spr_analysis_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        try:
            # Convert nested data to flat structure for CSV
            flat_data = self._flatten_data(data)
            
            # Create DataFrame
            if isinstance(flat_data, list):
                df = pd.DataFrame(flat_data)
            else:
                df = pd.DataFrame([flat_data])
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"Data exported to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.log_error(e, "CSV export failed")
            raise
    
    def export_to_excel(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                       filename: str = None) -> str:
        """
        Export data to Excel format with multiple sheets
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spr_analysis_{timestamp}.xlsx"
        
        filepath = self.output_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                
                if isinstance(data, list):
                    # Multiple companies comparison
                    # Summary sheet
                    summary_data = []
                    for company_data in data:
                        summary_row = {
                            'Company': company_data.get('company', 'Unknown'),
                            'Symbol': company_data.get('symbol', 'N/A'),
                            'SPR Score': company_data.get('spr_score', 0),
                            'Profit Performance': company_data.get('profit_performance', 0),
                            'Sustainability Impact': company_data.get('sustainability_impact', 0),
                            'Research Alignment': company_data.get('research_alignment', 0)
                        }
                        summary_data.append(summary_row)
                    
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Individual company sheets
                    for i, company_data in enumerate(data):
                        sheet_name = f"Company_{i+1}"
                        flat_data = self._flatten_data(company_data)
                        df = pd.DataFrame([flat_data])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                else:
                    # Single company analysis
                    # Main analysis sheet
                    main_data = {
                        'Metric': ['SPR Score', 'Profit Performance', 'Sustainability Impact', 
                                 'Research Alignment'],
                        'Value': [
                            data.get('spr_score', 0),
                            data.get('profit_performance', 0),
                            data.get('sustainability_impact', 0),
                            data.get('research_alignment', 0)
                        ]
                    }
                    pd.DataFrame(main_data).to_excel(writer, sheet_name='Analysis', index=False)
                    
                    # Financial metrics sheet
                    if 'financial_metrics' in data:
                        financial_df = pd.DataFrame([data['financial_metrics']])
                        financial_df.to_excel(writer, sheet_name='Financial_Metrics', index=False)
                    
                    # Research insights sheet
                    if 'research_insights' in data:
                        insights_df = pd.DataFrame({
                            'Research_Insights': data['research_insights']
                        })
                        insights_df.to_excel(writer, sheet_name='Research_Insights', index=False)
            
            self.logger.info(f"Data exported to Excel: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.log_error(e, "Excel export failed")
            raise
    
    def export_to_pdf(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Export data to PDF format
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF export requires reportlab package: pip install reportlab")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spr_analysis_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        try:
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("SPR Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Company information
            company_name = data.get('company', 'Unknown Company')
            story.append(Paragraph(f"<b>Company:</b> {company_name}", styles['Normal']))
            story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # SPR Scores Table
            scores_data = [
                ['Metric', 'Score', 'Rating'],
                ['SPR Score', f"{data.get('spr_score', 0):.2f}", self._get_rating(data.get('spr_score', 0))],
                ['Profit Performance', f"{data.get('profit_performance', 0):.2f}", self._get_rating(data.get('profit_performance', 0))],
                ['Sustainability Impact', f"{data.get('sustainability_impact', 0):.2f}", self._get_rating(data.get('sustainability_impact', 0))],
                ['Research Alignment', f"{data.get('research_alignment', 0):.2f}", self._get_rating(data.get('research_alignment', 0))]
            ]
            
            scores_table = Table(scores_data)
            scores_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("<b>SPR Analysis Scores</b>", styles['Heading2']))
            story.append(scores_table)
            story.append(Spacer(1, 20))
            
            # Financial Metrics
            if 'financial_metrics' in data:
                story.append(Paragraph("<b>Financial Metrics</b>", styles['Heading2']))
                for key, value in data['financial_metrics'].items():
                    if isinstance(value, (int, float)):
                        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value:,.2f}", styles['Normal']))
                    else:
                        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Key Findings
            if 'findings' in data and data['findings']:
                story.append(Paragraph("<b>Key Findings</b>", styles['Heading2']))
                for finding in data['findings'][:5]:  # Limit to top 5
                    story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Recommendations
            if 'recommendations' in data and data['recommendations']:
                story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
                for rec in data['recommendations'][:5]:  # Limit to top 5
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Data exported to PDF: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.log_error(e, "PDF export failed")
            raise
    
    def _flatten_data(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested dictionary for CSV export
        
        Args:
            data: Nested dictionary
            prefix: Prefix for keys
            
        Returns:
            Dict: Flattened dictionary
        """
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_data(value, new_key))
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    # Join string lists
                    flattened[new_key] = '; '.join(str(item) for item in value)
                else:
                    flattened[new_key] = str(value)
            else:
                flattened[new_key] = value
        
        return flattened
    
    def _get_rating(self, score: float) -> str:
        """
        Convert numeric score to rating
        
        Args:
            score: Numeric score (0-10)
            
        Returns:
            str: Rating description
        """
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"


class DataImporter(LoggerMixin):
    """Handles importing historical data and configurations"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data importer
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"DataImporter initialized with data directory: {self.data_dir}")
    
    def import_from_json(self, filepath: str) -> Dict[str, Any]:
        """
        Import data from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dict: Imported data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Data imported from JSON: {filepath}")
            return data
            
        except Exception as e:
            self.log_error(e, f"JSON import failed for {filepath}")
            raise
    
    def import_from_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Import data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List[Dict]: Imported data as list of dictionaries
        """
        try:
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
            
            self.logger.info(f"Data imported from CSV: {filepath} ({len(data)} records)")
            return data
            
        except Exception as e:
            self.log_error(e, f"CSV import failed for {filepath}")
            raise
    
    def import_from_excel(self, filepath: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        Import data from Excel file
        
        Args:
            filepath: Path to Excel file
            sheet_name: Specific sheet to import (None for all sheets)
            
        Returns:
            Dict: Imported data
        """
        try:
            if sheet_name:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                data = df.to_dict('records')
            else:
                excel_data = pd.read_excel(filepath, sheet_name=None)
                data = {}
                for sheet, df in excel_data.items():
                    data[sheet] = df.to_dict('records')
            
            self.logger.info(f"Data imported from Excel: {filepath}")
            return data
            
        except Exception as e:
            self.log_error(e, f"Excel import failed for {filepath}")
            raise
    
    def import_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Import historical analysis data for a company
        
        Args:
            symbol: Company symbol
            
        Returns:
            List[Dict]: Historical analysis data
        """
        try:
            # Look for historical data files
            pattern = f"{symbol.lower()}_*.json"
            files = list(self.data_dir.glob(pattern))
            
            historical_data = []
            for file in files:
                data = self.import_from_json(str(file))
                historical_data.append(data)
            
            # Sort by timestamp if available
            historical_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            self.logger.info(f"Imported {len(historical_data)} historical records for {symbol}")
            return historical_data
            
        except Exception as e:
            self.log_error(e, f"Historical data import failed for {symbol}")
            raise
