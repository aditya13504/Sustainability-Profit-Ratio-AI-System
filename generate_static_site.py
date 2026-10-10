"""
Generate static HTML version of SPR Dashboard for Netlify deployment
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
from pathlib import Path

print("Starting static site generation...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

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
    
    spr_chart_json = json.dumps(json.loads(spr_chart.to_json()), ensure_ascii=False)
    profit_chart_json = json.dumps(json.loads(profit_chart.to_json()), ensure_ascii=False)
    
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
        .data-notice {{
            border: 1px solid #c9d8ff;
            border-radius: 10px;
            background: #f3f6ff;
            color: #31436b;
            padding: 14px 18px;
            margin-bottom: 22px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px;
            margin: 20px 0 26px;
        }}
        .summary-card {{
            border: 1px solid #e9eaf2;
            border-radius: 10px;
            background: #fff;
            padding: 17px 20px;
        }}
        .summary-card span {{
            display: block;
            color: #6c757d;
            font-size: .9rem;
        }}
        .summary-card strong {{
            display: block;
            margin-top: 5px;
            color: #2c3e50;
            font-size: 1.55rem;
        }}
        .data-controls {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            margin: 18px 0;
        }}
        .data-controls p {{ margin: 0; color: #6c757d; }}
        .chart-error {{ display: none; }}
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
                        SPR Insights
                    </h1>
                    <p class="lead">Explore sustainability and profitability indicators</p>
                </div>
            </div>

            <div class="data-notice" role="status">
                <strong>Illustrative sample data</strong> — this static demo does not fetch live company, ESG, or market data, and does not generate AI assessments. Import your own CSV to explore it in this browser.
            </div>

            <section class="summary-grid" aria-label="Dataset summary">
                <div class="summary-card"><span>Companies shown</span><strong id="company-count">—</strong></div>
                <div class="summary-card"><span>Average SPR score</span><strong id="average-score">—</strong></div>
                <div class="summary-card"><span>Average profit margin</span><strong id="average-margin">—</strong></div>
            </section>

            <section class="data-controls" aria-label="Import company data">
                <p>CSV columns: company_name, symbol, sector, spr_score, sustainability_rating, profit_margin</p>
                <label class="btn btn-primary mb-0">
                    Import CSV
                    <input id="csv-file" type="file" accept=".csv,text/csv" hidden>
                </label>
            </section>
            <div id="data-message" class="alert d-none" role="status" aria-live="polite"></div>
            
            <!-- Charts -->
            <div class="row">
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">SPR Scores by Company</h4>
                        </div>
                        <div class="card-body">
                            <div id="spr-chart" role="img" aria-label="SPR score chart"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Profit Margin Analysis</h4>
                        </div>
                        <div class="card-body">
                            <div id="profit-chart" role="img" aria-label="Profit margin and SPR score chart"></div>
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
                        Demo analysis only · No live financial or sustainability data
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        const sampleCompanies = {data.to_json(orient="records")};
        const initialSprFigure = {spr_chart_json};
        const initialProfitFigure = {profit_chart_json};
        let companies = sampleCompanies;

        function showMessage(message, isError = false) {{
            const element = document.getElementById('data-message');
            element.textContent = message;
            element.className = `alert ${{isError ? 'alert-danger' : 'alert-success'}}`;
        }}

        function updateSummary() {{
            const averageScore = companies.reduce((sum, company) => sum + Number(company.spr_score), 0) / companies.length;
            const averageMargin = companies.reduce((sum, company) => sum + Number(company.profit_margin), 0) / companies.length;
            document.getElementById('company-count').textContent = companies.length.toLocaleString();
            document.getElementById('average-score').textContent = averageScore.toFixed(1);
            document.getElementById('average-margin').textContent = `${{(averageMargin * 100).toFixed(1)}}%`;

            const table = document.querySelector('.table tbody');
            table.replaceChildren(...companies.map((company) => {{
                const row = document.createElement('tr');
                [company.company_name, company.symbol, company.sector, Number(company.spr_score).toFixed(1),
                 company.sustainability_rating, `${{(Number(company.profit_margin) * 100).toFixed(1)}}%`]
                    .forEach((value) => {{
                        const cell = document.createElement('td');
                        cell.textContent = value;
                        row.appendChild(cell);
                    }});
                return row;
            }}));

            Plotly.react('spr-chart', [{{
                type: 'bar',
                x: companies.map((company) => company.company_name),
                y: companies.map((company) => Number(company.spr_score)),
                marker: {{ color: '#667eea' }},
                hovertemplate: '%{{x}}<br>SPR score: %{{y:.1f}}<extra></extra>'
            }}], {{ title: 'Sustainability–profit ratio score', yaxis: {{ title: 'Score', range: [0, 10] }}, margin: {{ t: 54, b: 100, l: 48, r: 20 }} }}, {{ responsive: true, displaylogo: false }});
            Plotly.react('profit-chart', [{{
                type: 'scatter',
                mode: 'markers',
                x: companies.map((company) => Number(company.profit_margin) * 100),
                y: companies.map((company) => Number(company.spr_score)),
                text: companies.map((company) => company.company_name),
                marker: {{ size: 14, color: '#764ba2', opacity: .8 }},
                hovertemplate: '%{{text}}<br>Profit margin: %{{x:.1f}}%<br>SPR score: %{{y:.1f}}<extra></extra>'
            }}], {{ title: 'Profit margin vs. SPR score', xaxis: {{ title: 'Profit margin (%)' }}, yaxis: {{ title: 'SPR score', range: [0, 10] }}, margin: {{ t: 54, b: 58, l: 48, r: 20 }} }}, {{ responsive: true, displaylogo: false }});
        }}

        function parseCsv(text) {{
            const rows = [];
            let row = [];
            let cell = '';
            let quoted = false;
            for (let index = 0; index < text.length; index += 1) {{
                const character = text[index];
                if (quoted) {{
                    if (character === '"' && text[index + 1] === '"') {{ cell += '"'; index += 1; }}
                    else if (character === '"') quoted = false;
                    else cell += character;
                }} else if (character === '"' && cell.length === 0) quoted = true;
                else if (character === ',') {{ row.push(cell); cell = ''; }}
                else if (character === '\\n' || character === '\\r') {{
                    if (character === '\\r' && text[index + 1] === '\\n') index += 1;
                    row.push(cell); cell = '';
                    if (row.some((value) => value.trim())) rows.push(row);
                    row = [];
                }} else cell += character;
            }}
            if (quoted) throw new Error('The CSV contains an unclosed quoted field.');
            row.push(cell);
            if (row.some((value) => value.trim())) rows.push(row);
            if (rows.length < 2) throw new Error('Add a header row and at least one company.');
            const headers = rows[0].map((value) => value.trim().toLowerCase());
            const required = ['company_name', 'symbol', 'sector', 'spr_score', 'sustainability_rating', 'profit_margin'];
            const indexes = required.map((name) => headers.indexOf(name));
            if (indexes.some((index) => index < 0)) throw new Error(`Required CSV headers: ${{required.join(', ')}}.`);
            return rows.slice(1).map((values, rowIndex) => {{
                const company = Object.fromEntries(required.map((name, index) => [name, (values[indexes[index]] || '').trim()]));
                const score = Number(company.spr_score);
                const marginInput = company.profit_margin;
                const margin = Number(marginInput.replace('%', '')) / (marginInput.endsWith('%') ? 100 : 1);
                if (!company.company_name || !company.symbol || !company.sector || !company.sustainability_rating ||
                    !Number.isFinite(score) || score < 0 || score > 10 || !Number.isFinite(margin) || margin < -1 || margin > 1) {{
                    throw new Error(`Invalid company data on CSV row ${{rowIndex + 2}}.`);
                }}
                return {{ ...company, spr_score: score, profit_margin: margin }};
            }});
        }}

        Plotly.newPlot('spr-chart', initialSprFigure.data, initialSprFigure.layout, {{ responsive: true, displaylogo: false }});
        Plotly.newPlot('profit-chart', initialProfitFigure.data, initialProfitFigure.layout, {{ responsive: true, displaylogo: false }});
        updateSummary();

        document.getElementById('csv-file').addEventListener('change', async (event) => {{
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            if (file.size > 5 * 1024 * 1024) {{
                showMessage('Choose a CSV file smaller than 5 MB.', true);
                event.target.value = '';
                return;
            }}
            try {{
                companies = parseCsv(await file.text());
                updateSummary();
                showMessage(`Loaded ${{companies.length}} companies from ${{file.name}} locally. No file data was uploaded.`);
            }} catch (error) {{
                showMessage(error.message || 'Could not read this CSV file.', true);
            }} finally {{
                event.target.value = '';
            }}
        }});
    </script>
</body>
</html>
    """
    
    return html_content

def main():
    """Main function to generate static site"""
    try:
        print("Creating dist directory...")
        # Create dist directory
        dist_dir = Path("dist")
        dist_dir.mkdir(exist_ok=True)
        print(f"✓ Directory created: {dist_dir.absolute()}")
        
        print("Generating HTML content...")
        # Generate HTML
        html_content = generate_html()
        print("✓ HTML content generated")
        
        print("Saving HTML file...")
        # Save HTML file
        with open(dist_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("✓ index.html saved")
        
        print("Creating data file...")
        # Create data file for potential future use
        data = create_sample_data()
        data.to_json(dist_dir / "data.json", orient="records", indent=2)
        print("✓ data.json saved")
        
        print("Static site generated successfully!")
        print(f"Files created in: {dist_dir.absolute()}")
        
        # List generated files
        print("\nGenerated files:")
        for file in dist_dir.iterdir():
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
            
    except Exception as e:
        print(f"Error during site generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
