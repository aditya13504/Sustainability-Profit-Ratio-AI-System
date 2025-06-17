"""
Simplified SPR Dashboard for Vercel Deployment
Minimal version with core functionality and sample data
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path

class SimplifiedSPRDashboard:
    """Simplified SPR Dashboard for Vercel deployment"""
    
    def __init__(self):
        """Initialize the simplified dashboard"""
        # Load sample data
        self.sample_data = self._load_sample_data()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            ]
        )
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_sample_data(self):
        """Load sample company data"""
        sample_data = [
            {
                "company_name": "Apple Inc.",
                "symbol": "AAPL",
                "sector": "Technology",
                "spr_score": 8.5,
                "sustainability_rating": "A",
                "profit_margin": 0.25,
                "description": "Leading technology company with strong sustainability initiatives"
            },
            {
                "company_name": "Microsoft Corp.",
                "symbol": "MSFT", 
                "sector": "Technology",
                "spr_score": 8.2,
                "sustainability_rating": "A",
                "profit_margin": 0.23,
                "description": "Cloud computing leader with carbon negative commitment"
            },
            {
                "company_name": "Johnson & Johnson",
                "symbol": "JNJ",
                "sector": "Healthcare", 
                "spr_score": 7.8,
                "sustainability_rating": "B+",
                "profit_margin": 0.18,
                "description": "Healthcare giant with focus on sustainable practices"
            },
            {
                "company_name": "Tesla Inc.",
                "symbol": "TSLA",
                "sector": "Automotive",
                "spr_score": 9.1,
                "sustainability_rating": "A+",
                "profit_margin": 0.15,
                "description": "Electric vehicle pioneer driving sustainable transportation"
            },
            {
                "company_name": "Unilever PLC",
                "symbol": "UL",
                "sector": "Consumer Goods",
                "spr_score": 7.5,
                "sustainability_rating": "A-",
                "profit_margin": 0.12,
                "description": "Consumer goods company with Sustainable Living Plan"
            }
        ]
        return pd.DataFrame(sample_data)
    
    def _setup_layout(self):
        """Create the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-chart-line me-3"),
                        "Enhanced SPR Analyzer"
                    ], className="text-center mb-4 text-primary")
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Analysis Settings", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Number of Companies:"),
                                    dcc.Dropdown(
                                        id="num-companies-dropdown",
                                        options=[
                                            {"label": "Top 5 Companies", "value": 5},
                                            {"label": "Top 10 Companies", "value": 10},
                                            {"label": "All Companies", "value": -1}
                                        ],
                                        value=5,
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Filter by Sector:"),
                                    dcc.Dropdown(
                                        id="sector-filter",
                                        options=[
                                            {"label": "All Sectors", "value": "all"},
                                            {"label": "Technology", "value": "Technology"},
                                            {"label": "Healthcare", "value": "Healthcare"},
                                            {"label": "Automotive", "value": "Automotive"},
                                            {"label": "Consumer Goods", "value": "Consumer Goods"}
                                        ],
                                        value="all",
                                        className="mb-3"
                                    )
                                ], width=6)
                            ]),
                            dbc.Button(
                                "Analyze Companies",
                                id="analyze-btn",
                                color="primary",
                                className="w-100"
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Results
            dbc.Row([
                dbc.Col([
                    html.Div(id="analysis-results")
                ])
            ])
        ], fluid=True, className="p-4")
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        
        @self.app.callback(
            Output("analysis-results", "children"),
            [Input("analyze-btn", "n_clicks")],
            [State("num-companies-dropdown", "value"),
             State("sector-filter", "value")]
        )
        def update_analysis(n_clicks, num_companies, sector_filter):
            if not n_clicks:
                return html.Div([
                    dbc.Alert(
                        "Click 'Analyze Companies' to view the analysis results.",
                        color="info",
                        className="text-center"
                    )
                ])
            
            # Filter data
            filtered_data = self.sample_data.copy()
            if sector_filter != "all":
                filtered_data = filtered_data[filtered_data["sector"] == sector_filter]
            
            # Sort by SPR score
            filtered_data = filtered_data.sort_values("spr_score", ascending=False)
            
            # Limit number of companies
            if num_companies > 0:
                filtered_data = filtered_data.head(num_companies)
            
            # Create visualizations
            spr_chart = self._create_spr_chart(filtered_data)
            profit_chart = self._create_profit_chart(filtered_data)
            company_table = self._create_company_table(filtered_data)
            
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("SPR Scores by Company"),
                            dcc.Graph(figure=spr_chart)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Profit Margin Analysis"),
                            dcc.Graph(figure=profit_chart)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Company Details"),
                            company_table
                        ])
                    ], className="mt-3")
                ], width=12)
            ])
    
    def _create_spr_chart(self, data):
        """Create SPR score chart"""
        fig = px.bar(
            data,
            x="company_name",
            y="spr_score",
            color="sustainability_rating",
            title="Sustainability-Profit Ratio Scores",
            labels={"spr_score": "SPR Score", "company_name": "Company"}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig
    
    def _create_profit_chart(self, data):
        """Create profit margin chart"""
        fig = px.scatter(
            data,
            x="profit_margin",
            y="spr_score",
            size="profit_margin",
            color="sector",
            hover_name="company_name",
            title="SPR Score vs Profit Margin",
            labels={"profit_margin": "Profit Margin", "spr_score": "SPR Score"}
        )
        fig.update_layout(height=400)
        return fig
    
    def _create_company_table(self, data):
        """Create company details table"""
        table_data = []
        for _, row in data.iterrows():
            table_data.append(
                html.Tr([
                    html.Td(row["company_name"]),
                    html.Td(row["symbol"]),
                    html.Td(row["sector"]),
                    html.Td(f"{row['spr_score']:.1f}"),
                    html.Td(row["sustainability_rating"]),
                    html.Td(f"{row['profit_margin']:.1%}")
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Company"),
                    html.Th("Symbol"),
                    html.Th("Sector"),
                    html.Th("SPR Score"),
                    html.Th("Sustainability Rating"),
                    html.Th("Profit Margin")
                ])
            ]),
            html.Tbody(table_data)
        ], striped=True, bordered=True, hover=True, responsive=True)

# For Vercel deployment
dashboard = SimplifiedSPRDashboard()
app = dashboard.app.server

if __name__ == "__main__":
    dashboard.app.run_server(debug=True)
