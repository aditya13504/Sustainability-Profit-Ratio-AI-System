# Create minimal sample data for Vercel deployment
import pandas as pd
import json
from pathlib import Path

# Sample company data
sample_companies = [
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

# Create sample DataFrame
df = pd.DataFrame(sample_companies)

# Save as CSV
output_dir = Path("sample_data")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "sample_companies.csv", index=False)

# Save as JSON
with open(output_dir / "sample_companies.json", "w") as f:
    json.dump(sample_companies, f, indent=2)

print("Sample data created successfully!")
