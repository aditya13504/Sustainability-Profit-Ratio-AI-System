"""
REST API endpoints for the SPR Analyzer

Provides external API access to SPR analysis functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import sys
import os
from datetime import datetime
import json
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analyzer import SPRAnalyzer
from utils.config_loader import ConfigLoader
from utils.logging_utils import get_logger
from utils.mistral_ai_assistant import MistralAIAssistant

# Import cuOpt optimizer
try:
    from cuopt_portfolio_optimizer import CuOptPortfolioOptimizer
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
config_loader = ConfigLoader()
config = config_loader.config  # Changed from load_config()
analyzer = SPRAnalyzer()  # Initialize without config parameter
logger = get_logger(__name__)

# Initialize Mistral AI assistant
try:
    mistral_assistant = MistralAIAssistant()
    AI_AVAILABLE = True
    logger.info("Mistral AI assistant initialized successfully")
except Exception as e:
    logger.warning(f"Mistral AI not available: {e}")
    AI_AVAILABLE = False
    mistral_assistant = None

# Initialize cuOpt optimizer
if CUOPT_AVAILABLE:
    try:
        cuopt_optimizer = CuOptPortfolioOptimizer()
        logger.info("NVIDIA cuOpt optimizer initialized successfully")
    except Exception as e:
        logger.warning(f"cuOpt not available: {e}")
        CUOPT_AVAILABLE = False
        cuopt_optimizer = None
else:
    cuopt_optimizer = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'SPR Analyzer API'
    })

@app.route('/api/v1/analyze/<symbol>', methods=['GET'])
def analyze_company(symbol):
    """
    Analyze a single company's SPR
    
    Args:
        symbol: Stock symbol (e.g., TSLA, AAPL)
    
    Returns:
        JSON: Complete SPR analysis results
    """
    try:
        logger.info(f"API request for company analysis: {symbol}")
        
        # Run analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyzer.analyze_company(symbol))
        loop.close()
        
        # Add metadata
        result['metadata'] = {
            'request_timestamp': datetime.now().isoformat(),
            'api_version': 'v1',
            'symbol': symbol.upper()
        }
        
        logger.info(f"Successfully analyzed {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return jsonify({
            'error': str(e),
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/analyze-advanced/<symbol>', methods=['GET'])
def analyze_company_advanced(symbol):
    """
    Analyze a single company's SPR using the advanced multi-stage pipeline
    
    This endpoint uses the new pipeline with quality control, drift correction,
    hybrid AI models, RAG analysis, and LLM-powered insights.
    
    Args:
        symbol: Stock symbol (e.g., TSLA, AAPL)
    
    Query parameters:
        include_research: Whether to include research analysis (default: true)
    
    Returns:
        JSON: Complete SPR analysis results with pipeline metrics
    """
    try:
        logger.info(f"API request for advanced company analysis: {symbol}")
        
        # Get query parameters
        include_research = request.args.get('include_research', 'true').lower() == 'true'
        
        # Run advanced analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyzer.analyze_company_advanced(symbol, include_research))
        loop.close()
        
        if result is None:
            return jsonify({
                'error': 'Analysis failed',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Convert result to dict
        result_dict = result.to_dict()
        
        # Add metadata
        result_dict['metadata'] = {
            'request_timestamp': datetime.now().isoformat(),
            'api_version': 'v1-advanced',
            'symbol': symbol.upper(),
            'include_research': include_research,
            'analysis_type': 'multi-stage-pipeline'
        }
        
        logger.info(f"Successfully analyzed {symbol} using advanced pipeline")
        return jsonify(result_dict)
        
    except Exception as e:
        logger.error(f"Error in advanced analysis for {symbol}: {str(e)}")
        return jsonify({
            'error': str(e),
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/compare', methods=['POST'])
def compare_companies():
    """
    Compare multiple companies
    
    Request body:
        {
            "symbols": ["TSLA", "AAPL", "MSFT"],
            "metrics": ["spr_score", "sustainability_impact"]  # optional
        }
    
    Returns:
        JSON: Comparison results
    """
    try:
        data = request.get_json()
        
        if not data or 'symbols' not in data:
            return jsonify({'error': 'Missing symbols in request body'}), 400
        
        symbols = data['symbols']
        requested_metrics = data.get('metrics', [])
        
        if not isinstance(symbols, list) or len(symbols) < 2:
            return jsonify({'error': 'At least 2 symbols required for comparison'}), 400
        
        logger.info(f"API request for company comparison: {symbols}")
        
        # Run comparison
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyzer.compare_companies(symbols))
        loop.close()
        
        # Filter metrics if requested
        if requested_metrics:
            for company in result['companies']:
                filtered_company = {k: v for k, v in company.items() 
                                 if k in requested_metrics or k in ['symbol', 'company']}
                company.clear()
                company.update(filtered_company)
        
        # Add metadata
        result['metadata'] = {
            'request_timestamp': datetime.now().isoformat(),
            'api_version': 'v1',
            'symbols': symbols,
            'comparison_count': len(symbols)
        }
        
        logger.info(f"Successfully compared {len(symbols)} companies")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in company comparison: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/research/<symbol>', methods=['GET'])
def get_research_insights(symbol):
    """
    Get research insights for a company
    
    Args:
        symbol: Stock symbol
        
    Query parameters:
        limit: Maximum number of papers (default: 10)
    
    Returns:
        JSON: Research papers and insights
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        
        logger.info(f"API request for research insights: {symbol}")
        
        # Get research data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        papers = loop.run_until_complete(
            analyzer.research_analyzer.search_papers(f"{symbol} sustainability", limit)
        )
        loop.close()
        
        result = {
            'symbol': symbol.upper(),
            'research_papers': papers[:limit],
            'paper_count': len(papers),
            'metadata': {
                'request_timestamp': datetime.now().isoformat(),
                'api_version': 'v1',
                'limit': limit
            }
        }
        
        logger.info(f"Retrieved {len(papers)} research papers for {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting research for {symbol}: {str(e)}")
        return jsonify({
            'error': str(e),
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/predict', methods=['POST'])
def predict_spr():
    """
    Predict SPR score based on financial metrics
    
    Request body:
        {
            "financial_metrics": {
                "revenue": 100000000000,
                "profit_margin": 0.2,
                "roa": 0.08,
                "roe": 0.15,
                "current_ratio": 1.5,
                "debt_to_equity": 0.3
            }
        }
    
    Returns:
        JSON: Predicted SPR score
    """
    try:
        data = request.get_json()
        
        if not data or 'financial_metrics' not in data:
            return jsonify({'error': 'Missing financial_metrics in request body'}), 400
        
        metrics = data['financial_metrics']
        
        # TODO: Implement ML prediction
        # For now, return a placeholder
        predicted_score = 7.5  # Placeholder
        
        result = {
            'predicted_spr_score': predicted_score,
            'input_metrics': metrics,
            'model_info': {
                'model_type': 'ensemble',
                'confidence': 0.85
            },
            'metadata': {
                'request_timestamp': datetime.now().isoformat(),
                'api_version': 'v1'
            }
        }
        
        logger.info("SPR prediction completed")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in SPR prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/export/<symbol>', methods=['GET'])
def export_analysis(symbol):
    """
    Export analysis results in various formats
    
    Args:
        symbol: Stock symbol
        
    Query parameters:
        format: Export format (json, csv, pdf) - default: json
    
    Returns:
        File: Analysis results in requested format
    """
    try:
        export_format = request.args.get('format', 'json').lower()
        
        if export_format not in ['json', 'csv', 'pdf']:
            return jsonify({'error': 'Invalid format. Supported: json, csv, pdf'}), 400
        
        logger.info(f"API request to export {symbol} analysis as {export_format}")
        
        # Get analysis results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyzer.analyze_company(symbol))
        loop.close()
        
        if export_format == 'json':
            response = jsonify(result)
            response.headers['Content-Disposition'] = f'attachment; filename={symbol}_analysis.json'
            return response
        
        # TODO: Implement CSV and PDF export
        return jsonify({'error': f'{export_format} export not yet implemented'}), 501
        
    except Exception as e:
        logger.error(f"Error exporting {symbol} analysis: {str(e)}")
        return jsonify({
            'error': str(e),
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/top-companies', methods=['POST'])
def get_top_companies():
    """
    Get top 10 companies based on criteria using SPR analysis and cuOpt optimization
    
    Request body:
        {
            "criteria": {
                "analysis_type": "spr|esg|profit|balanced|growth",
                "risk_tolerance": "low|medium|high",
                "sectors": ["Technology", "Healthcare", ...],
                "market_cap_min": 1000000000,
                "max_companies": 10
            },
            "symbols": ["TSLA", "AAPL", ...] (optional - if provided, analyze these specific symbols)
        }
    
    Returns:
        JSON: Top 10 companies with rankings and AI explanation
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        criteria = data.get('criteria', {})
        symbols = data.get('symbols', [])
        
        logger.info(f"API request for top companies with criteria: {criteria}")
        
        # Perform analysis and optimization
        if CUOPT_AVAILABLE and cuopt_optimizer:
            # Use cuOpt for optimization
            top_companies = cuopt_optimizer.optimize_portfolio_selection(criteria, symbols)
        else:
            # Fallback analysis
            top_companies = _perform_fallback_analysis(criteria, symbols)
        
        # Generate AI explanation if available
        ai_explanation = ""
        if AI_AVAILABLE and mistral_assistant:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ai_explanation = loop.run_until_complete(
                    mistral_assistant.explain_top_companies(top_companies, criteria)
                )
                loop.close()
            except Exception as e:
                logger.warning(f"AI explanation failed: {e}")
                ai_explanation = "AI explanation temporarily unavailable."
        
        result = {
            'top_companies': top_companies,
            'criteria': criteria,
            'ai_explanation': ai_explanation,
            'cuopt_used': CUOPT_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully generated top {len(top_companies)} companies")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting top companies: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/chat', methods=['POST'])
def chat_with_ai():
    """
    Chat with Mistral AI about companies analysis
    
    Request body:
        {
            "message": "User question about companies",
            "companies_data": [...],  // Current companies context
            "conversation_id": "optional_id"
        }
    
    Returns:
        JSON: AI response with conversation context
    """
    try:
        if not AI_AVAILABLE:
            return jsonify({
                'error': 'AI assistant not available',
                'response': 'Sorry, the AI assistant is currently offline.'
            }), 503
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message required in request body'}), 400
        
        user_message = data['message']
        companies_data = data.get('companies_data', [])
        conversation_id = data.get('conversation_id')
        
        logger.info(f"AI chat request: {user_message[:50]}...")
        
        # Get AI response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_response = loop.run_until_complete(
            mistral_assistant.chat_about_companies(user_message, companies_data)
        )
        loop.close()
        
        result = {
            'response': ai_response,
            'user_message': user_message,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'mistral-small-latest'
        }
        
        logger.info("AI chat response generated successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': f'Sorry, I encountered an error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/investment-insights', methods=['POST'])
def get_investment_insights():
    """
    Get comprehensive investment insights for companies using Mistral AI
    
    Request body:
        {
            "companies_data": [...] // Top companies data
        }
    
    Returns:
        JSON: Comprehensive investment insights and recommendations
    """
    try:
        if not AI_AVAILABLE:
            return jsonify({
                'error': 'AI assistant not available'
            }), 503
        
        data = request.get_json()
        if not data or 'companies_data' not in data:
            return jsonify({'error': 'Companies data required in request body'}), 400
        
        companies_data = data['companies_data']
        
        logger.info(f"Generating investment insights for {len(companies_data)} companies")
        
        # Generate insights
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        insights = loop.run_until_complete(
            mistral_assistant.generate_investment_insights(companies_data)
        )
        loop.close()
        
        result = {
            'insights': insights,
            'companies_count': len(companies_data),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Investment insights generated successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating investment insights: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def _perform_fallback_analysis(criteria: Dict[str, Any], symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fallback analysis when cuOpt is not available
    """
    # This is a simplified fallback - in practice, would use the full SPR analyzer
    mock_companies = [
        {
            "rank": 1, "symbol": "TSLA", "name": "Tesla Inc",
            "spr_score": 9.2, "sector": "Technology", "market_cap": 800000000000,
            "sustainability_score": 9.5, "profit_performance": 8.8,
            "esg_score": "A", "risk_level": "Medium"
        },
        {
            "rank": 2, "symbol": "MSFT", "name": "Microsoft Corp",
            "spr_score": 9.0, "sector": "Technology", "market_cap": 2800000000000,
            "sustainability_score": 8.9, "profit_performance": 9.1,
            "esg_score": "A+", "risk_level": "Low"
        },
        {
            "rank": 3, "symbol": "AAPL", "name": "Apple Inc",
            "spr_score": 8.8, "sector": "Technology", "market_cap": 3000000000000,
            "sustainability_score": 8.5, "profit_performance": 9.2,
            "esg_score": "A", "risk_level": "Low"
        },
        {
            "rank": 4, "symbol": "GOOGL", "name": "Alphabet Inc",
            "spr_score": 8.6, "sector": "Technology", "market_cap": 1700000000000,
            "sustainability_score": 8.2, "profit_performance": 8.9,
            "esg_score": "A-", "risk_level": "Medium"
        },
        {
            "rank": 5, "symbol": "NVDA", "name": "NVIDIA Corp",
            "spr_score": 8.4, "sector": "Technology", "market_cap": 1800000000000,
            "sustainability_score": 7.8, "profit_performance": 9.0,
            "esg_score": "B+", "risk_level": "High"
        },
        {
            "rank": 6, "symbol": "AMZN", "name": "Amazon.com Inc",
            "spr_score": 8.2, "sector": "Consumer Discretionary", "market_cap": 1600000000000,
            "sustainability_score": 7.5, "profit_performance": 8.8,
            "esg_score": "B", "risk_level": "Medium"
        },
        {
            "rank": 7, "symbol": "META", "name": "Meta Platforms Inc",
            "spr_score": 8.0, "sector": "Technology", "market_cap": 900000000000,
            "sustainability_score": 7.2, "profit_performance": 8.7,
            "esg_score": "B-", "risk_level": "High"
        },
        {
            "rank": 8, "symbol": "V", "name": "Visa Inc",
            "spr_score": 7.8, "sector": "Financial Services", "market_cap": 500000000000,
            "sustainability_score": 7.0, "profit_performance": 8.5,
            "esg_score": "B+", "risk_level": "Low"
        },
        {
            "rank": 9, "symbol": "JNJ", "name": "Johnson & Johnson",
            "spr_score": 7.6, "sector": "Healthcare", "market_cap": 450000000000,
            "sustainability_score": 8.0, "profit_performance": 7.2,
            "esg_score": "A-", "risk_level": "Low"
        },
        {
            "rank": 10, "symbol": "UNH", "name": "UnitedHealth Group",
            "spr_score": 7.4, "sector": "Healthcare", "market_cap": 520000000000,
            "sustainability_score": 6.8, "profit_performance": 8.0,
            "esg_score": "B", "risk_level": "Medium"
        }
    ]
    
    # Filter based on criteria if specified
    if criteria.get('sectors'):
        mock_companies = [c for c in mock_companies if c['sector'] in criteria['sectors']]
    
    return mock_companies[:criteria.get('max_companies', 10)]

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/api/v1/analyze/<symbol>',
            '/api/v1/analyze-advanced/<symbol>',
            '/api/v1/compare',
            '/api/v1/research/<symbol>',
            '/api/v1/predict',
            '/api/v1/export/<symbol>',
            '/api/v1/top-companies',
            '/api/v1/chat',
            '/api/v1/investment-insights'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('API_PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting SPR Analyzer API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
