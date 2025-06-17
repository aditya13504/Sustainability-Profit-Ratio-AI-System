"""
Generate static HTML version of SPR Dashboard for Netlify deployment
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from pathlib import Path

def create_sample_data():
    """Create sample company data"""
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

def create_charts(data):
    """Create Plotly charts"""
    # SPR Score Chart
    spr_chart = px.bar(
        data,
        x="company_name",
        y="spr_score",
        color="sustainability_rating",
        title="Sustainability-Profit Ratio Scores",
        labels={"spr_score": "SPR Score", "company_name": "Company"}
    )
    spr_chart.update_layout(height=400, xaxis_tickangle=-45)
    
    # Profit Margin Chart
    profit_chart = px.scatter(
        data,
        x="profit_margin",
        y="spr_score",
        size="profit_margin",
        color="sector",
        hover_name="company_name",
        title="SPR Score vs Profit Margin",
        labels={"profit_margin": "Profit Margin", "spr_score": "SPR Score"}
    )
    profit_chart.update_layout(height=400)
    
    return spr_chart, profit_chart

def generate_html():
    """Generate static HTML file"""
    data = create_sample_data()
    spr_chart, profit_chart = create_charts(data)
    
    # Convert charts to HTML
    spr_chart_html = spr_chart.to_html(include_plotlyjs='cdn', div_id="spr-chart")
    profit_chart_html = profit_chart.to_html(include_plotlyjs='cdn', div_id="profit-chart")
    
    # Create company table HTML
    table_rows = ""
    for _, row in data.iterrows():
        table_rows += f"""
        <tr>
            <td>{row['company_name']}</td>
            <td>{row['symbol']}</td>
            <td>{row['sector']}</td>
            <td>{row['spr_score']:.1f}</td>
            <td>{row['sustainability_rating']}</td>
            <td>{row['profit_margin']:.1%}</td>
        </tr>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced SPR Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .main-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            padding: 30px;
        }}
        .card {{
            border: none;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
        }}
        .card-header {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 10px 10px 0 0 !important;
        }}
        .btn-primary {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 8px;
        }}
        .table {{
            border-radius: 8px;
            overflow: hidden;
        }}
        h1 {{
            color: #2c3e50;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="main-container">
            <!-- Header -->
            <div class="row">
                <div class="col-12 text-center mb-4">
                    <h1>
                        <i class="fas fa-chart-line me-3"></i>
                        Enhanced SPR Analyzer
                    </h1>
                    <p class="lead">Sustainability-Profit Ratio Analysis Dashboard</p>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">SPR Scores by Company</h4>
                        </div>
                        <div class="card-body">
                            <div id="spr-chart"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Profit Margin Analysis</h4>
                        </div>
                        <div class="card-body">
                            <div id="profit-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Company Table -->
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Company Details</h4>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped table-hover">
                                    <thead class="table-dark">
                                        <tr>
                                            <th>Company</th>
                                            <th>Symbol</th>
                                            <th>Sector</th>
                                            <th>SPR Score</th>
                                            <th>Sustainability Rating</th>
                                            <th>Profit Margin</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {table_rows}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="row mt-4">
                <div class="col-12 text-center">
                    <p class="text-muted">
                        <i class="fas fa-leaf me-2"></i>
                        Powered by AI-driven sustainability analysis
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        // Extract and render SPR chart
        {spr_chart.to_json()}
        // Extract and render profit chart  
        {profit_chart.to_json()}
    </script>
</body>
</html>
    """
    
    return html_content

def main():
    """Main function to generate static site"""
    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Generate HTML
    html_content = generate_html()
    
    # Save HTML file
    with open(dist_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Create data file for potential future use
    data = create_sample_data()
    data.to_json(dist_dir / "data.json", orient="records", indent=2)
    
    print("Static site generated successfully!")
    print(f"Files created in: {dist_dir.absolute()}")

if __name__ == "__main__":
    main()
