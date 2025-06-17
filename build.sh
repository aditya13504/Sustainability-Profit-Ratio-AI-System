#!/bin/bash
# Build script for Netlify deployment

echo "Starting Enhanced SPR Analyzer build process..."

# Check Python version
echo "Python version:"
python --version

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Test dependencies
echo "Testing dependencies..."
python test_dependencies.py

# Generate static site
echo "Generating static site..."
python generate_static_site.py

echo "Build completed successfully!"
