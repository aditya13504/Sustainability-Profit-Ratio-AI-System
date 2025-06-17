"""
Vercel deployment entry point for Enhanced SPR Dashboard
Simplified version with minimal dependencies
"""
import sys
import os
from pathlib import Path

# Import the simplified dashboard
from app_simplified import SimplifiedSPRDashboard

# Initialize the simplified dashboard
dashboard = SimplifiedSPRDashboard()

# Vercel expects a Flask/Django app or WSGI application
# Since this is a Dash app, we need to expose the Flask server
app = dashboard.app.server

if __name__ == "__main__":
    # For local testing
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
