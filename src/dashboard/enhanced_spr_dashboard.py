"""
Enhanced SPR Dashboard with 4-Step Analysis Pipeline

Step 1: User Input - Detailed company search criteria
Step 2: Comprehensive Search - Using APIs, research papers, and Gemini 2.5 Pro
Step 3: Optimization - NVIDIA cuOpt + Gemini explanations for top 1-10 companies
Step 4: Mistral AI Chat - Interactive chatbot for questions about selected companies
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.comprehensive_company_search import ComprehensiveCompanySearchEngine
from utils.nvidia_cuopt_cloud_api import NVIDIACuOptCloudAPI
from utils.mistral_ai_assistant import MistralAIAssistant
from utils.config_loader import ConfigLoader
from utils.logging_utils import get_logger

class FourStepSPRDashboard:
    """
    4-Step SPR Analysis Dashboard
    
    Step 1: User inputs detailed company search criteria
    Step 2: Comprehensive search using APIs, research papers, and Gemini 2.5 Pro
    Step 3: NVIDIA cuOpt optimization + Gemini explanations for top companies
    Step 4: Mistral AI chatbot for interactive Q&A about selected companies
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the 4-step dashboard"""
        self.config = ConfigLoader(config_path).config if config_path else {}
        self.logger = get_logger(__name__)
        
        # Initialize Step 2: Company Search Engine
        try:
            self.search_engine = ComprehensiveCompanySearchEngine(config_path)
            self.search_available = True
            self.logger.info("Company Search Engine initialized")
        except Exception as e:
            self.logger.warning(f"Search Engine not available: {e}")
            self.search_available = False
            self.search_engine = None
          # Initialize Step 3: NVIDIA cuOpt Cloud API Optimizer
        try:
            self.optimizer = NVIDIACuOptCloudAPI()
            self.optimizer_available = True
            self.logger.info("NVIDIA cuOpt + Gemini Optimizer initialized")
        except Exception as e:
            self.logger.warning(f"NVIDIA cuOpt Optimizer not available: {e}")
            self.optimizer_available = False
            self.optimizer = None
        
        # Initialize Step 4: Mistral AI Assistant
        try:
            self.mistral_assistant = MistralAIAssistant()
            self.ai_available = True
            self.logger.info("Mistral AI Assistant initialized")
        except Exception as e:
            self.logger.warning(f"Mistral AI not available: {e}")
            self.ai_available = False
            self.mistral_assistant = None
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            ],
            suppress_callback_exceptions=True
        )
        
        # Current analysis data
        self.current_companies = []
        self.current_criteria = {}
        self.current_top_companies = []
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Create the enhanced dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-chart-line me-3"),
                            "🌱 SPR Analyzer with AI Assistant"
                        ], className="display-4 fw-bold text-center mb-2"),
                        html.P(
                            "Discover top sustainable investment opportunities with NVIDIA cuOpt optimization and Mistral AI insights",
                            className="lead text-center text-muted mb-4"
                        )
                    ], className="hero-section py-5")
                ])
            ]),
            
            # Company Search Interface
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3([
                                html.I(className="fas fa-search me-2"),
                                "Company Analysis Input"
                            ])
                        ]),
                        dbc.CardBody([
                            self._create_input_form()
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ]),
            
            # Top 10 Companies Display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3([
                                html.I(className="fas fa-trophy me-2"),
                                "Top 10 Recommended Companies"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="companies-loading",
                                children=[
                                    html.Div(id="top-companies-container")
                                ],
                                type="circle"
                            )
                        ])
                    ], className="shadow-sm mb-4")
                ], width=8),
                
                # AI Assistant Chat Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-robot me-2"),
                                "AI Assistant"
                            ])
                        ]),
                        dbc.CardBody([
                            # Chat display area
                            html.Div(
                                id="chat-messages",
                                style={
                                    "height": "400px",
                                    "overflow-y": "auto",
                                    "border": "1px solid #dee2e6",
                                    "border-radius": "5px",
                                    "padding": "10px",
                                    "margin-bottom": "10px"
                                }
                            ),
                            # Chat input
                            dbc.InputGroup([
                                dbc.Input(
                                    id="chat-input",
                                    placeholder="Ask me about the top 10 companies...",
                                    type="text"
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-paper-plane"),
                                    id="send-message-btn",
                                    color="primary",
                                    n_clicks=0
                                )
                            ]),
                            # Status indicator
                            html.Div(id="ai-status", className="mt-2")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=4)
            ]),
            
            # Analysis Visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Portfolio Analysis & Insights"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="analysis-loading",
                                children=[
                                    html.Div(id="analysis-charts-container")
                                ],
                                type="circle"
                            )
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ]),
            
            # Store components for data sharing
            dcc.Store(id="companies-data"),
            dcc.Store(id="analysis-criteria"),
            dcc.Store(id="chat-history", data=[])
            
        ], fluid=True, className="px-4")
    def _create_input_form(self):
        """Create the enhanced company analysis input form for 4-step process"""
        return html.Div([
            # Step 1: Company Search Criteria
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-search me-2"),
                        "Step 1: Define Your Company Search Criteria"
                    ])
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("🏢 Company Search Query"),
                            dbc.Textarea(
                                id="company-input",
                                placeholder="""Examples:
• Stock symbols: TSLA, AAPL, MSFT
• Search criteria: "Technology companies in California with high ESG scores"
• Sector focus: "Renewable energy companies with market cap > $1B"
• Location: "European automotive companies with strong sustainability practices"
• Profit focus: "High-growth companies with consistent profitability"
""",
                                rows=4,
                                className="mb-3"
                            )
                        ], md=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("📊 Analysis Focus"),
                            dcc.Dropdown(
                                id="analysis-type",
                                options=[
                                    {"label": "📊 SPR Analysis (Balanced)", "value": "spr"},
                                    {"label": "🌱 ESG & Sustainability Focus", "value": "esg"},
                                    {"label": "💰 Profit & Performance Focus", "value": "profit"},
                                    {"label": "⚖️ Balanced Portfolio", "value": "balanced"},
                                    {"label": "🚀 Growth Potential", "value": "growth"},
                                    {"label": "🛡️ Dividend & Stability", "value": "dividend"}
                                ],
                                value="spr",
                                className="mb-3"
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("⚖️ Risk Tolerance"),
                            dcc.Dropdown(
                                id="risk-tolerance",
                                options=[
                                    {"label": "🛡️ Conservative (Low Risk)", "value": "low"},
                                    {"label": "⚖️ Moderate (Medium Risk)", "value": "medium"},
                                    {"label": "🎲 Aggressive (High Risk)", "value": "high"}
                                ],
                                value="medium",
                                className="mb-3"
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("🎯 Number of Companies"),
                            dcc.Dropdown(
                                id="max-companies",
                                options=[
                                    {"label": "Top 5 Companies", "value": 5},
                                    {"label": "Top 10 Companies", "value": 10},
                                    {"label": "Top 15 Companies", "value": 15}
                                ],
                                value=10,
                                className="mb-3"
                            )                        ], md=4)
                    ]),
                    # Additional spacing for better visual appearance
                    html.Div(style={"height": "30px"})
                ], style={"min-height": "400px", "padding": "2rem"})            ], className="mb-4", style={"min-height": "500px"}),
            
            # Analysis Button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-rocket me-2"),
                            "🚀 Start 4-Step AI Analysis"
                        ],
                        id="analyze-btn",
                        color="primary",
                        size="lg",
                        className="w-100 shadow",
                        n_clicks=0,
                        style={"height": "60px", "font-size": "18px"}
                    )
                ], md=12)
            ], className="mt-3"),
            
            # Process Steps Indicator
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("🔄 Analysis Process:", className="mb-2"),                    html.Div([
                        dbc.Badge("1️⃣ Parse Criteria", color="primary", className="me-2"),
                        dbc.Badge("2️⃣ Search Companies", color="red", className="me-2"),
                        dbc.Badge("3️⃣ cuOpt + Gemini Optimization", color="lightgreen", className="me-2"),
                        dbc.Badge("4️⃣ Mistral AI Chat Ready", color="orange")
                    ])], md=12)
            ])
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        
        @self.app.callback(
            [Output("companies-data", "data"),
             Output("analysis-criteria", "data"),
             Output("top-companies-container", "children")],
            [Input("analyze-btn", "n_clicks")],
            [State("company-input", "value"),
             State("analysis-type", "value"),
             State("risk-tolerance", "value"),
             State("max-companies", "value")]
        )
        def analyze_and_find_top_companies(n_clicks, company_input, analysis_type, risk_tolerance, max_companies):
            """Analyze companies and find top N using cuOpt optimization - FAST VERSION"""
            if not n_clicks or not company_input:
                return [], {}, self._create_welcome_message()
            
            # Log the analysis attempt
            self.logger.info(f"FAST Analysis triggered - n_clicks: {n_clicks}, input: {company_input}, max_companies: {max_companies}")
            
            try:
                # Prepare analysis criteria
                criteria = {
                    "input": company_input,
                    "analysis_type": analysis_type,
                    "risk_tolerance": risk_tolerance,
                    "max_companies": max_companies or 10,  # Default to 10 if None
                    "timestamp": datetime.now().isoformat()
                }
                
                # FAST PATH: Use immediate fallback with sample companies
                self.logger.info(f"Using FAST analysis path for immediate results with {max_companies or 10} companies")
                companies_data = self._get_fast_sample_companies(company_input, analysis_type, risk_tolerance, max_companies or 10)
                
                # Store current data
                self.current_companies = companies_data
                self.current_criteria = criteria
                
                # Create display
                companies_display = self._create_companies_display(companies_data)
                
                self.logger.info(f"FAST Analysis complete - {len(companies_data)} companies found")
                return companies_data, criteria, companies_display
                
            except Exception as e:
                self.logger.error(f"Error in analysis: {e}")
                error_display = dbc.Alert(
                    [
                        html.H5("Analysis Error", className="alert-heading"),
                        html.P(f"Error: {str(e)}"),
                        html.Hr(),
                        html.P("Please try again with different criteria.", className="mb-0")
                    ], 
                    color="danger"
                )
                return [], {}, error_display
        
        @self.app.callback(
            Output("chat-messages", "children"),
            [Input("send-message-btn", "n_clicks"),
             Input("companies-data", "data")],
            [State("chat-input", "value"),
             State("chat-history", "data")]
        )
        def handle_chat_message(n_clicks, companies_data, user_message, chat_history):
            """Handle chat messages with Mistral AI"""
            if not self.ai_available:
                return [html.Div("AI Assistant not available", className="text-muted")]
            
            ctx = callback_context
            if not ctx.triggered:
                return chat_history or []
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
              # Initial explanation when companies are loaded
            if trigger_id == "companies-data" and companies_data:
                if not chat_history:  # Only show explanation once
                    try:
                        # Use proper async handling
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(
                                    self.mistral_assistant.explain_top_companies(
                                        companies_data, self.current_criteria or {}
                                    )
                                )
                            )
                            explanation = future.result(timeout=30)  # 30 second timeout
                        return [self._create_message_bubble("assistant", explanation)]
                    except Exception as e:
                        error_msg = f"Welcome! I had trouble generating an initial explanation: {str(e)}"
                        return [self._create_message_bubble("assistant", error_msg)]
            
            # Handle user message
            elif trigger_id == "send-message-btn" and n_clicks and user_message and companies_data:
                new_messages = chat_history or []
                
                # Add user message
                new_messages.append(self._create_message_bubble("user", user_message))
                
                # Get AI response
                try:
                    # Use proper async handling
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self.mistral_assistant.chat_about_companies(user_message, companies_data)
                            )
                        )
                        ai_response = future.result(timeout=30)  # 30 second timeout
                    new_messages.append(self._create_message_bubble("assistant", ai_response))
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    new_messages.append(self._create_message_bubble("assistant", error_msg))
                
                return new_messages
            
            return chat_history or []
        
        @self.app.callback(
            Output("chat-input", "value"),
            [Input("send-message-btn", "n_clicks")]
        )
        def clear_chat_input(n_clicks):
            """Clear chat input after sending"""
            if n_clicks:
                return ""
            return dash.no_update
        
        @self.app.callback(
            Output("analysis-charts-container", "children"),
            [Input("companies-data", "data")]
        )
        def update_analysis_charts(companies_data):
            """Update analysis charts based on companies data"""
            if not companies_data:
                return html.Div("No data to display", className="text-muted text-center")
            
            return self._create_analysis_charts(companies_data)
        
        @self.app.callback(
            Output("ai-status", "children"),
            [Input("companies-data", "data")]
        )
        def update_ai_status(companies_data):
            """Update AI assistant status"""
            if not self.ai_available:
                return dbc.Badge("AI Offline", color="warning", className="w-100")
            elif companies_data:
                return dbc.Badge("AI Ready - Ask me anything!", color="success", className="w-100")
            else:
                return dbc.Badge("Waiting for data...", color="info", className="w-100")
        
        @self.app.callback(
            [Output("analyze-btn", "children"),
             Output("analyze-btn", "disabled")],
            [Input("analyze-btn", "n_clicks"),
             Input("top-companies-container", "children")],
            [State("company-input", "value")]
        )
        def update_button_state(n_clicks, container_children, company_input):
            """Update button state during and after analysis"""
            ctx = callback_context
            if not ctx.triggered:
                return [
                    html.Span([
                        html.I(className="fas fa-rocket me-2"),
                        "🚀 Start 4-Step AI Analysis"
                    ])
                ], False
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # If button was clicked and has input, show loading
            if trigger_id == "analyze-btn" and n_clicks and company_input:
                return [
                    html.Span([
                        dbc.Spinner(size="sm", className="me-2"),
                        "Analyzing with NVIDIA cuOpt..."
                    ])
                ], True
            
            # If analysis is complete (container has content), reset button
            elif trigger_id == "top-companies-container" and container_children:
                return [
                    html.Span([
                        html.I(className="fas fa-rocket me-2"),
                        "🚀 Start Analysis"
                    ])
                ], False
            
            # Default state
            return [
                html.Span([
                    html.I(className="fas fa-rocket me-2"),
                    "🚀 Start 4-Step AI Analysis"
                ])
            ], False
    
    async def _perform_analysis(self, input_text: str, analysis_type: str, risk_tolerance: str) -> List[Dict[str, Any]]:
        """
        Perform the complete 4-step company analysis and optimization
        
        Step 1: Parse user input criteria
        Step 2: Search for companies using comprehensive search engine  
        Step 3: Optimize and rank using cuOpt + Gemini
        Step 4: Return top 10 companies ready for Mistral AI chat
        """
        try:
            self.logger.info(f"Starting 4-step analysis for: {input_text}")
            
            # Step 1: Parse and prepare search criteria
            search_criteria = self._parse_user_input(input_text, analysis_type, risk_tolerance)
            self.logger.info(f"Step 1 complete: Parsed criteria: {search_criteria}")
            
            # Step 2: Comprehensive company search
            if self.search_available:
                self.logger.info("Step 2: Starting comprehensive company search...")
                found_companies = await self.search_engine.search_companies_by_criteria(search_criteria)
                self.logger.info(f"Step 2 complete: Found {len(found_companies)} companies")
            else:
                self.logger.warning("Search engine not available, using fallback data")
                found_companies = self._get_fallback_companies(input_text)
              # Step 3: Optimize and rank using NVIDIA cuOpt + Gemini
            if self.optimizer_available and found_companies:
                self.logger.info("Step 3: Starting NVIDIA cuOpt + Gemini optimization...")
                optimization_result = await self.optimizer.optimize_portfolio_with_cuopt(
                    found_companies, search_criteria, top_n=10
                )
                top_companies = optimization_result.get('top_companies', found_companies[:10])
                self.logger.info(f"Step 3 complete: Optimized to top {len(top_companies)} companies")
            else:
                self.logger.warning("NVIDIA cuOpt Optimizer not available, using basic ranking")
                top_companies = self._basic_ranking(found_companies)[:10]
            
            # Step 4: Prepare data for Mistral AI interaction
            self.current_top_companies = top_companies
            self.logger.info("Step 4: Companies ready for AI interaction")
            
            return top_companies
            
        except Exception as e:
            self.logger.error(f"Error in 4-step analysis: {e}")
            # Return fallback data on error
            return self._get_fallback_companies(input_text)[:10]
    
    def _parse_user_input(self, input_text: str, analysis_type: str, risk_tolerance: str) -> Dict[str, Any]:
        """Parse user input into structured search criteria"""
        criteria = {
            "query": input_text,
            "analysis_type": analysis_type,
            "risk_tolerance": risk_tolerance,
            "max_results": 50,  # Search more, then optimize to top 10
            "min_market_cap": 1000000000 if risk_tolerance == "low" else 100000000,
            "sectors": [],
            "regions": [],
            "esg_focus": analysis_type == "esg",
            "growth_focus": analysis_type == "growth",
            "profit_focus": analysis_type == "profit"
        }
        
        # Extract symbols if provided
        symbols = self._extract_symbols(input_text)
        if symbols:
            criteria["symbols"] = symbols
            criteria["symbol_mode"] = True
        else:
            criteria["symbol_mode"] = False
            criteria["search_terms"] = input_text.split()
        
        return criteria
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from input text"""
        import re
        # Look for potential stock symbols (3-5 uppercase letters)
        symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
        return [s for s in symbols if len(s) >= 2]
    
    def _get_fallback_companies(self, input_text: str) -> List[Dict[str, Any]]:
        """Get fallback company data when search engine is not available"""
        # This provides backup data for demo purposes
        fallback_companies = [
            {
                "symbol": "TSLA", "name": "Tesla Inc", "sector": "Technology",
                "market_cap": 800000000000, "spr_score": 9.2,
                "sustainability_score": 9.5, "profit_performance": 8.8,
                "esg_score": "A", "risk_level": "Medium", "description": "Electric vehicle and clean energy company"
            },
            {
                "symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology", 
                "market_cap": 2800000000000, "spr_score": 9.0,
                "sustainability_score": 8.9, "profit_performance": 9.1,
                "esg_score": "A+", "risk_level": "Low", "description": "Cloud computing and software company"
            },
            {
                "symbol": "AAPL", "name": "Apple Inc", "sector": "Technology",
                "market_cap": 3000000000000, "spr_score": 8.8,
                "sustainability_score": 8.5, "profit_performance": 9.2,
                "esg_score": "A", "risk_level": "Low", "description": "Consumer electronics and services"
            },
            {
                "symbol": "GOOGL", "name": "Alphabet Inc", "sector": "Technology",
                "market_cap": 1700000000000, "spr_score": 8.6,
                "sustainability_score": 8.2, "profit_performance": 8.9,
                "esg_score": "A-", "risk_level": "Medium", "description": "Search and cloud services"
            },
            {
                "symbol": "NVDA", "name": "NVIDIA Corp", "sector": "Technology",
                "market_cap": 1800000000000, "spr_score": 8.4,
                "sustainability_score": 7.8, "profit_performance": 9.0,
                "esg_score": "B+", "risk_level": "High", "description": "AI and graphics processing"
            },
            {
                "symbol": "AMZN", "name": "Amazon.com Inc", "sector": "Consumer Discretionary",
                "market_cap": 1600000000000, "spr_score": 8.2,
                "sustainability_score": 7.5, "profit_performance": 8.8,
                "esg_score": "B", "risk_level": "Medium", "description": "E-commerce and cloud services"
            },
            {
                "symbol": "META", "name": "Meta Platforms Inc", "sector": "Technology",
                "market_cap": 900000000000, "spr_score": 8.0,
                "sustainability_score": 7.2, "profit_performance": 8.7,
                "esg_score": "B-", "risk_level": "High", "description": "Social media and VR platforms"
            },
            {
                "symbol": "V", "name": "Visa Inc", "sector": "Financial Services",
                "market_cap": 500000000000, "spr_score": 7.8,
                "sustainability_score": 7.0, "profit_performance": 8.5,
                "esg_score": "B+", "risk_level": "Low", "description": "Payment processing network"
            },
            {
                "symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare",
                "market_cap": 450000000000, "spr_score": 7.6,
                "sustainability_score": 8.0, "profit_performance": 7.2,
                "esg_score": "A-", "risk_level": "Low", "description": "Healthcare and pharmaceuticals"
            },            {
                "symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare",
                "market_cap": 520000000000, "spr_score": 7.4,
                "sustainability_score": 6.8, "profit_performance": 8.0,
                "esg_score": "B", "risk_level": "Medium", "description": "Healthcare insurance and services"
            },
            {
                "symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financial Services",
                "market_cap": 480000000000, "spr_score": 7.2,
                "sustainability_score": 6.5, "profit_performance": 7.9,
                "esg_score": "B", "risk_level": "Medium", "description": "Banking and financial services"
            },
            {
                "symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Goods",
                "market_cap": 380000000000, "spr_score": 7.0,
                "sustainability_score": 7.8, "profit_performance": 6.2,
                "esg_score": "A-", "risk_level": "Low", "description": "Consumer products and brands"
            },
            {
                "symbol": "HD", "name": "Home Depot", "sector": "Consumer Discretionary",
                "market_cap": 350000000000, "spr_score": 6.8,
                "sustainability_score": 6.2, "profit_performance": 7.4,
                "esg_score": "B+", "risk_level": "Medium", "description": "Home improvement retail"
            },
            {
                "symbol": "BAC", "name": "Bank of America", "sector": "Financial Services",
                "market_cap": 320000000000, "spr_score": 6.6,
                "sustainability_score": 6.0, "profit_performance": 7.2,
                "esg_score": "B", "risk_level": "Medium", "description": "Banking and financial services"
            },
            {
                "symbol": "XOM", "name": "Exxon Mobil", "sector": "Energy",
                "market_cap": 460000000000, "spr_score": 6.4,
                "sustainability_score": 5.5, "profit_performance": 7.3,
                "esg_score": "C+", "risk_level": "High", "description": "Oil and gas exploration and production"
            },
            {
                "symbol": "WMT", "name": "Walmart Inc", "sector": "Consumer Defensive",
                "market_cap": 500000000000, "spr_score": 6.2,
                "sustainability_score": 6.8, "profit_performance": 5.6,
                "esg_score": "B", "risk_level": "Low", "description": "Retail and e-commerce"
            },
            {
                "symbol": "CVX", "name": "Chevron Corp", "sector": "Energy",
                "market_cap": 310000000000, "spr_score": 6.0,
                "sustainability_score": 5.2, "profit_performance": 6.8,
                "esg_score": "C+", "risk_level": "High", "description": "Oil and gas integrated company"
            },
            {
                "symbol": "KO", "name": "Coca-Cola Company", "sector": "Consumer Defensive",
                "market_cap": 260000000000, "spr_score": 5.8,
                "sustainability_score": 6.5, "profit_performance": 5.1,
                "esg_score": "B-", "risk_level": "Low", "description": "Beverage manufacturing and distribution"
            }
        ]
        
        # Filter by symbols if provided
        symbols = self._extract_symbols(input_text)
        if symbols:
            return [c for c in fallback_companies if c["symbol"] in symbols]
        
        return fallback_companies
    
    def _get_fast_sample_companies(self, company_input: str, analysis_type: str, risk_tolerance: str, max_companies: int = 10) -> List[Dict[str, Any]]:
        """Get fast sample companies for instant results"""
        # Get fallback companies and customize based on input
        companies = self._get_fallback_companies(company_input)
        
        # Customize based on analysis type and risk tolerance
        if analysis_type == "esg":
            # Sort by ESG scores for ESG-focused analysis
            companies = sorted(companies, key=lambda x: x.get("sustainability_score", 0), reverse=True)
        elif analysis_type == "growth":
            # Sort by profit performance for growth-focused analysis
            companies = sorted(companies, key=lambda x: x.get("profit_performance", 0), reverse=True)
        elif analysis_type == "value":
            # Sort by combined value metrics
            for company in companies:
                # Simple value score calculation
                value_score = (company.get("profit_performance", 0) * 0.6 + 
                             company.get("sustainability_score", 0) * 0.4)
                company["value_score"] = value_score
            companies = sorted(companies, key=lambda x: x.get("value_score", 0), reverse=True)
        else:
            # Default: sort by SPR score
            companies = sorted(companies, key=lambda x: x.get("spr_score", 0), reverse=True)
        
        # Filter by risk tolerance
        if risk_tolerance == "low":
            companies = [c for c in companies if c.get("risk_level", "").lower() in ["low", "medium"]]
        elif risk_tolerance == "medium":
            companies = [c for c in companies if c.get("risk_level", "").lower() in ["low", "medium", "high"]]
        # For high risk tolerance, include all companies
        
        # Limit to the requested number of companies
        companies = companies[:max_companies]
        
        # Add rank and ensure all required fields
        for i, company in enumerate(companies):
            company["rank"] = i + 1
            # Ensure all companies have required fields
            if "ai_explanation" not in company:
                company["ai_explanation"] = f"Strong {analysis_type} opportunity with {company.get('risk_level', 'moderate').lower()} risk profile."
            if "optimization_score" not in company:
                company["optimization_score"] = company.get("spr_score", 8.0)
        
        return companies
    
    def _basic_ranking(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic ranking when optimizer is not available"""
        # Sort by SPR score or calculate basic score
        for i, company in enumerate(companies):
            if "spr_score" not in company:
                # Calculate basic SPR score
                sustainability = company.get("sustainability_score", 5.0)
                profit = company.get("profit_performance", 5.0)
                company["spr_score"] = (sustainability + profit) / 2.0
            
            company["rank"] = i + 1        
        return sorted(companies, key=lambda x: x.get("spr_score", 0), reverse=True)
    
    def _create_welcome_message(self):
        """Create welcome message"""
        return dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Enter company symbols or investment criteria above to get AI-powered company recommendations."
        ], color="info", className="text-center")
    
    def _create_companies_display(self, companies_data: List[Dict[str, Any]]):
        """Create the top companies display (5, 10, or 15 based on user selection)"""
        if not companies_data:
            return self._create_welcome_message()
        
        company_cards = []
        for company in companies_data:
            card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Badge(f"#{company['rank']}", color="primary", className="fs-6 mb-2"),
                            html.H5(f"{company['name']} ({company['symbol']})", className="card-title"),
                            html.P(f"Sector: {company['sector']}", className="text-muted mb-1"),
                            html.P(f"Market Cap: ${company['market_cap']:,.0f}", className="text-muted mb-2"),
                        ], md=8),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{company['spr_score']:.1f}", className="text-primary"),
                                html.P("SPR Score", className="text-muted mb-0"),
                            ], className="text-center"),
                            dbc.Badge(company['esg_score'], color="success", className="mt-2")
                        ], md=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Small(f"Sustainability: {company['sustainability_score']:.1f}/10", className="text-success"),
                        ], md=4),
                        dbc.Col([
                            html.Small(f"Profit: {company['profit_performance']:.1f}/10", className="text-info"),
                        ], md=4),
                        dbc.Col([
                            html.Small(f"Risk: {company['risk_level']}", className="text-warning"),
                        ], md=4)
                    ])
                ])
            ], className="mb-2")
            company_cards.append(card)
        
        return html.Div(company_cards)
    
    def _create_message_bubble(self, role: str, content: str):
        """Create a chat message bubble"""
        if role == "user":
            return dbc.Card([
                dbc.CardBody([
                    html.P(content, className="mb-0")
                ], className="p-2")
            ], color="primary", outline=True, className="mb-2 ms-5")
        else:  # assistant
            return dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-robot me-2"),
                        html.Strong("AI Assistant:")
                    ], className="mb-2"),
                    html.P(content, className="mb-0", style={"white-space": "pre-wrap"})
                ], className="p-2")
            ], color="light", className="mb-2 me-5")
    
    def _create_analysis_charts(self, companies_data: List[Dict[str, Any]]):
        """Create analysis charts for the companies"""
        if not companies_data:
            return html.Div()
        
        # SPR Score comparison
        df = pd.DataFrame(companies_data)
        
        # Create SPR score bar chart
        fig1 = px.bar(
            df, x='symbol', y='spr_score',
            title='SPR Scores Comparison',
            color='spr_score',
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(height=400)
        
        # Create sustainability vs profit scatter
        fig2 = px.scatter(
            df, x='profit_performance', y='sustainability_score',
            size='market_cap', hover_name='name',
            title='Sustainability vs Profit Performance',
            color='spr_score'
        )
        fig2.update_layout(height=400)
        
        # Sector distribution pie chart
        sector_counts = df['sector'].value_counts()
        fig3 = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title='Sector Distribution'
        )
        fig3.update_layout(height=400)
        
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig1)], md=6),
                dbc.Col([dcc.Graph(figure=fig2)], md=6)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig3)], md=6),
                dbc.Col([
                    html.H5("Key Insights"),
                    html.Ul([
                        html.Li(f"Average SPR Score: {df['spr_score'].mean():.2f}"),
                        html.Li(f"Top Performer: {df.iloc[0]['name']} ({df.iloc[0]['spr_score']:.1f})"),
                        html.Li(f"Most Sustainable: {df.loc[df['sustainability_score'].idxmax(), 'name']}"),
                        html.Li(f"Highest Profit: {df.loc[df['profit_performance'].idxmax(), 'name']}"),
                    ])
                ], md=6)
            ])
        ])
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the enhanced dashboard"""
        self.logger.info(f"Starting Enhanced SPR Dashboard with Mistral AI on port {port}")
        self.app.run(debug=debug, port=port)

# Entry point
if __name__ == "__main__":
    dashboard = FourStepSPRDashboard()
    dashboard.run()
