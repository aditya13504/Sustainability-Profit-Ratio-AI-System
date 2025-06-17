"""
Vercel deployment entry point for Enhanced SPR Dashboard
"""
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the dashboard
from dashboard.enhanced_spr_dashboard import FourStepSPRDashboard

# Initialize the dashboard
dashboard = FourStepSPRDashboard()

# Vercel expects a Flask/Django app or WSGI application
# Since this is a Dash app, we need to expose the Flask server
app = dashboard.app.server

if __name__ == "__main__":
    # For local testing
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
