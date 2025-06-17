"""
Enhanced SPR Analyzer Startup Script

This script starts the 4-Step SPR Analyzer Dashboard with:
1. User Input for Company Search Criteria
2. Comprehensive Company Search (APIs + Research Papers + Gemini)
3. NVIDIA cuOpt + Gemini Optimization 
4. Mistral AI Chatbot Integration

Usage:
    python start_enhanced_spr_dashboard.py [--port 8050] [--debug]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from dashboard.enhanced_spr_dashboard import FourStepSPRDashboard
except ImportError as e:
    print(f"Error importing dashboard: {e}")
    print("Make sure all dependencies are installed with: pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "spr_dashboard.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ Warning: .env file not found!")
        print("Copy .env.template to .env and add your API keys for full functionality:")
        print("- GEMINI_API_KEY (for Step 2 & 3)")
        print("- MISTRAL_API_KEY (for Step 4 chatbot)")
        print("- Other API keys for enhanced data sources")
        print()
    
    # Check for required directories
    required_dirs = ["data", "logs", "outputs"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

def main():
    """Main function to start the enhanced dashboard"""
    parser = argparse.ArgumentParser(description="Start Enhanced SPR Analyzer Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    check_environment()
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Enhanced SPR Analyzer with 4-Step AI Process")
    
    try:
        # Initialize and run dashboard
        dashboard = FourStepSPRDashboard()
        
        print("\n" + "="*80)
        print("🌱 SPR ANALYZER - 4-STEP AI-POWERED INVESTMENT ANALYSIS")
        print("="*80)
        print(f"🌐 Dashboard URL: http://{args.host}:{args.port}")
        print("📋 PROCESS OVERVIEW:")
        print("   1️⃣ Input: Define your company search criteria")
        print("   2️⃣ Search: Multi-source company discovery (APIs + Research + Gemini)")
        print("   3️⃣ Optimize: NVIDIA cuOpt + Gemini explanations")
        print("   4️⃣ Chat: Mistral AI assistant for Q&A about top companies")
        print("\n🔧 FEATURES:")
        print("   • 🤖 Google Gemini 2.5 Pro integration")
        print("   • 🚀 NVIDIA cuOpt optimization")
        print("   • 💬 Mistral Mixtral 8×7B chatbot")
        print("   • 📊 Real-time financial data")
        print("   • 🌱 ESG and sustainability analysis")
        print("="*80)
          # Run the dashboard
        dashboard.app.run(
            debug=args.debug,
            host=args.host,
            port=args.port,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify API keys in .env file")
        print("3. Check logs/spr_dashboard.log for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
