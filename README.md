# Enhanced SPR Analyzer - Comprehensive Investment Analysis Platform

![SPR Analyzer](https://img.shields.io/badge/SPR-Analyzer-green?style=for-the-badge) ![AI Powered](https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.9+-orange?style=for-the-badge)

## 🌟 Overview

The **Enhanced SPR (Sustainability-Profit-Research) Analyzer** is a cutting-edge investment analysis platform that combines artificial intelligence, financial data analysis, and academic research to provide comprehensive company evaluations. This project implements a sophisticated 4-step AI-powered analysis pipeline that helps investors make informed decisions based on sustainability metrics, financial performance, and research-backed insights.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced SPR Analyzer                        │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: User Input → Step 2: AI Search → Step 3: Optimization  │
│                    → Step 4: AI Chat Assistant                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🤖 **4-Step AI Analysis Pipeline**

1. **Step 1: Intelligent Input Processing**
   - Advanced natural language understanding for investment criteria
   - Multi-dimensional search parameter extraction
   - Risk tolerance assessment and profiling

2. **Step 2: Comprehensive Company Discovery**
   - **Google Gemini 2.5 Pro** integration for intelligent company search
   - Multi-source data aggregation (APIs + Academic Research)
   - Real-time financial data collection from Yahoo Finance, Alpha Vantage, Finnhub
   - Global company registry integration (SEC, ASIC, Companies House, etc.)

3. **Step 3: NVIDIA cuOpt + AI Optimization**
   - **NVIDIA cuOpt** GPU-accelerated portfolio optimization
   - **Google Gemini** AI explanations for ranking decisions
   - Advanced SPR (Sustainability-Profit-Research) scoring algorithm
   - Multi-criteria decision analysis

4. **Step 4: Interactive AI Assistant**
   - **Mistral Mixtral 8×7B** chatbot for Q&A about selected companies
   - Real-time investment insights and recommendations
   - Context-aware conversation about portfolio analysis

### 📊 **Advanced Analytics & AI Models**

- **Hybrid AI Architecture**: CNNs + Transformers for financial time-series and text analysis
- **Retrieval-Augmented Generation (RAG)**: Research paper analysis with academic context
- **Predictive Models**: Machine learning forecasting for SPR scores and financial performance
- **Sentiment Analysis**: News and social media sentiment integration
- **Industry-Specific Metrics**: Tailored analysis for different business sectors

### 🌍 **Comprehensive Data Sources**

- **Financial APIs**: Yahoo Finance, Alpha Vantage, Finnhub, NewsAPI
- **Company Registries**: SEC EDGAR, UK Companies House, Australia ASIC, Canada Corporations, EU BRIS
- **Research Papers**: Automated academic paper analysis and insight extraction
- **ESG Data**: Environmental, Social, and Governance metrics integration
- **Market Data**: Real-time stock prices, historical data, and market indicators

## 🛠️ Technology Stack

### **Frontend & Dashboard**
- **Dash (Plotly)**: Interactive web dashboard with modern UI
- **Bootstrap Components**: Responsive design and professional styling
- **Plotly**: Advanced data visualization and interactive charts

### **Backend & AI**
- **Python 3.9+**: Core application framework
- **Flask**: RESTful API server
- **Google Gemini 2.5 Pro**: Advanced natural language processing
- **Mistral AI**: Conversational AI assistant
- **NVIDIA cuOpt**: GPU-accelerated optimization

### **Data & Analytics**
- **Pandas & NumPy**: Data processing and numerical analysis
- **Scikit-learn**: Machine learning models and algorithms
- **PyTorch**: Deep learning and neural networks
- **OpenPyXL**: Excel file processing and export

### **Infrastructure**
- **Vercel**: Serverless deployment platform
- **SQLite**: Local data storage and caching
- **aiohttp**: Asynchronous HTTP client for API calls

## 📁 Project Structure

```
research-profit-analyzer/
├── 📁 src/                          # Source code
│   ├── 📁 dashboard/                # Web dashboard
│   │   ├── enhanced_spr_dashboard.py    # Main 4-step dashboard
│   │   ├── visualizations.py           # Advanced charts & graphs
│   │   └── assets/                      # CSS, images, styling
│   ├── 📁 models/                   # AI & ML models
│   │   ├── advanced_spr_analyzer.py    # Enhanced SPR calculations
│   │   ├── hybrid_ai_model.py          # CNN + Transformer hybrid
│   │   ├── rag_analyzer.py             # Research paper RAG system
│   │   ├── predictive_models.py        # ML forecasting models
│   │   ├── pipeline_manager.py         # Multi-stage AI pipeline
│   │   └── quality_controller.py       # Data quality assurance
│   ├── 📁 data_collectors/          # Data acquisition
│   │   ├── enhanced_financial_collector.py  # Financial data APIs
│   │   ├── esg_data_collector.py           # ESG metrics collector
│   │   ├── company_registry_collector.py   # Global registries
│   │   └── research_paper_collector.py     # Academic papers
│   ├── 📁 utils/                    # Utility modules
│   │   ├── gemini_ai_assistant.py       # Google Gemini integration
│   │   ├── mistral_ai_assistant.py      # Mistral AI chatbot
│   │   ├── nvidia_cuopt_cloud_api.py    # NVIDIA cuOpt integration
│   │   ├── comprehensive_company_search.py  # Multi-source search
│   │   └── sentiment_analyzer.py        # News/social sentiment
│   ├── 📁 financial/                # Financial analysis
│   │   └── data_processor.py            # Financial metrics calculation
│   ├── 📁 research_processor/       # Academic research
│   │   └── paper_analyzer.py            # Research paper NLP
│   └── analyzer.py                  # Main SPR analyzer engine
├── 📁 data/                         # Data storage
│   ├── 📁 raw/                      # Raw financial & research data
│   │   ├── financial_reports/           # Company fundamentals
│   │   ├── market_data/                 # Stock prices & history
│   │   └── research_papers/             # Academic papers (20+ papers)
│   └── 📁 registry/                 # Company registry data
├── app.py                           # Vercel entry point
├── vercel.json                      # Vercel deployment config
├── requirements.txt                 # Python dependencies
├── start_enhanced_spr_dashboard.py # Local development server
└── README.md                        # Project documentation
```

## 🔧 Installation & Setup

### **Local Development**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aditya13504/Sustainability-Profit-Ratio-AI-System.git
   cd research-profit-analyzer
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run the Dashboard**
   ```bash
   python start_enhanced_spr_dashboard.py
   ```

5. **Access the Application**
   - Open your browser to `http://localhost:8050`
   - Follow the 4-step analysis process

### **Vercel Deployment**

1. **Fork this repository** to your GitHub account

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your forked repository

3. **Configure Environment Variables** in Vercel dashboard:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   FINNHUB_API_KEY=your_finnhub_key
   NEWSAPI_KEY=your_newsapi_key
   ```

4. **Deploy**: Vercel will automatically build and deploy your dashboard

## 🔑 Required API Keys

| Service | Purpose | Get Key From |
|---------|---------|--------------|
| **Google Gemini** | AI analysis & optimization explanations | [Google AI Studio](https://makersuite.google.com) |
| **Mistral AI** | Interactive chatbot assistant | [Mistral AI Console](https://console.mistral.ai) |
| **Alpha Vantage** | Financial data & fundamentals | [Alpha Vantage](https://www.alphavantage.co) |
| **Finnhub** | Real-time market data | [Finnhub](https://finnhub.io) |
| **NewsAPI** | News sentiment analysis | [NewsAPI](https://newsapi.org) |

## 🎯 Usage Guide

### **Step 1: Define Investment Criteria**
- Enter your investment goals and criteria in natural language
- Specify risk tolerance (Conservative, Moderate, Aggressive)
- Set analysis type (Growth, Value, ESG-focused, etc.)
- Choose number of companies to analyze (5, 10, or 15)

### **Step 2: AI-Powered Company Discovery**
- The system uses Google Gemini 2.5 Pro to understand your criteria
- Searches multiple data sources including:
  - Financial APIs for real-time data
  - Global company registries
  - Academic research papers
  - ESG databases

### **Step 3: NVIDIA cuOpt Optimization**
- GPU-accelerated portfolio optimization
- Advanced SPR scoring algorithm
- Google Gemini provides AI explanations for rankings
- Multi-criteria decision analysis results

### **Step 4: Interactive AI Assistant**
- Chat with Mistral AI about your selected companies
- Ask questions about financial performance, sustainability, risks
- Get personalized investment recommendations
- Export analysis results to Excel/PDF

## 📈 Analysis Metrics

### **SPR Score Components**
- **Profit Performance Score** (0-10): Financial health and profitability
- **Sustainability Impact Score** (0-10): ESG metrics and environmental impact
- **Research Alignment Score** (0-10): Academic research support
- **Risk Factor** (0-1): Overall investment risk assessment

### **Advanced Analytics**
- **Predictive Models**: 6-12 month performance forecasting
- **Sentiment Analysis**: News and social media sentiment scores
- **Industry Benchmarking**: Sector-specific comparisons
- **Correlation Analysis**: Cross-metric relationship mapping

## 🔬 Research Paper Integration

The system includes analysis of 20+ academic papers on sustainability and profitability:

- "Does it Pay to be Green?" - Environmental performance and financial returns
- "ESG in Corporate Filings: An AI Perspective" - ESG reporting analysis
- "Financial Markets and ESG" - Market impact of sustainability initiatives
- "Crosswashing in Sustainable Investing" - Investment strategy analysis
- And many more covering various aspects of sustainable finance

## 📊 Dashboard Features

### **Interactive Visualizations**
- **SPR Score Comparisons**: Bar charts and radar plots
- **Sustainability vs Profit Scatter**: Performance correlation
- **Sector Analysis**: Industry distribution and benchmarking
- **Risk-Return Analysis**: Portfolio optimization visualizations
- **Time Series**: Historical performance trends

### **Export Capabilities**
- Excel reports with detailed analysis
- PDF investment summaries
- JSON data exports for further analysis
- Chart image exports (PNG/SVG)

## 🔮 AI Model Details

### **Hybrid AI Architecture**
- **CNN Layers**: Process financial time-series data
- **Transformer Layers**: Analyze sustainability text data
- **Attention Mechanisms**: Focus on relevant features
- **Ensemble Methods**: Combine multiple model predictions

### **RAG (Retrieval-Augmented Generation)**
- **Knowledge Base**: 20+ research papers indexed
- **Semantic Search**: Find relevant research for queries
- **Context Generation**: Create research-backed insights
- **Fact Verification**: Cross-reference multiple sources

### **Quality Control Pipeline**
- **Data Validation**: Automated quality checks
- **Drift Detection**: Monitor data consistency
- **Confidence Scoring**: Reliability assessment
- **Error Handling**: Graceful degradation strategies

## 🚀 Performance & Scalability

### **Optimization Features**
- **GPU Acceleration**: NVIDIA cuOpt for portfolio optimization
- **Async Processing**: Non-blocking API calls and data processing
- **Caching System**: SQLite-based result caching
- **Batch Processing**: Efficient multi-company analysis
- **Rate Limiting**: API quota management

### **Scalability Considerations**
- **Serverless Architecture**: Vercel deployment for auto-scaling
- **Modular Design**: Component-based architecture
- **API Abstraction**: Easy integration of new data sources
- **Configuration Management**: YAML-based settings

## 🔒 Security & Privacy

- **API Key Management**: Secure environment variable storage
- **Data Encryption**: SQLite database encryption support
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Protection against API abuse
- **Error Handling**: Secure error messages without data leakage

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📋 Roadmap

### **Phase 1: Enhanced AI Integration**
- [ ] GPT-4 integration for advanced analysis
- [ ] Claude AI integration for risk assessment
- [ ] Expanded research paper database (100+ papers)

### **Phase 2: Advanced Features**
- [ ] Real-time portfolio monitoring
- [ ] ESG trend prediction models
- [ ] Social media sentiment tracking
- [ ] Custom alert system

### **Phase 3: Enterprise Features**
- [ ] Multi-user support and authentication
- [ ] Advanced export formats (PowerBI, Tableau)
- [ ] API rate limiting and quotas
- [ ] Enterprise dashboard themes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI** for Gemini 2.5 Pro API access
- **Mistral AI** for conversational AI capabilities
- **NVIDIA** for cuOpt optimization platform
- **Academic Researchers** for sustainability finance research
- **Open Source Community** for the amazing Python ecosystem