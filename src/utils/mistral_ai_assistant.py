"""
Mistral AI Assistant for SPR Analyzer

Provides AI-powered explanations and chatbot functionality using Mistral Mixtral 8×7B
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    Mistral = None

class MistralAIAssistant:
    """Mistral AI-powered assistant for SPR analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mistral AI assistant
        
        Args:
            api_key: Mistral API key (will use environment variable if not provided)
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI package not available. Install with: pip install mistralai")
        
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key or self.api_key == 'your_mistral_api_key_here':
            # Don't raise error if API key is missing, just log warning for demo mode
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Mistral API key not found. Demo mode only.")
            self.client = None
        else:
            try:
                self.client = Mistral(api_key=self.api_key)
                # Test the connection
                self.logger.info("Mistral AI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral client: {e}")
                self.client = None
        
        self.model = "mistral-small-latest"  # Using free tier model
        self.logger = logging.getLogger(__name__)
        
        # Maximum tokens for responses
        self.max_tokens = 4096
        
        # Conversation history
        self.conversation_history = []
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        
    async def explain_top_companies(self, companies_data: List[Dict[str, Any]], 
                                  selection_criteria: Dict[str, Any]) -> str:
        """
        Generate explanation for why these top 10 companies were selected
        
        Args:
            companies_data: List of top 10 companies with their data
            selection_criteria: Criteria used for selection
            
        Returns:
            str: Detailed explanation of the selection
        """
        if not self.client:
            return "AI explanation not available - Mistral API key required. Demo mode active."
        
        try:
            # Prepare company summary
            company_summaries = []
            for i, company in enumerate(companies_data[:10], 1):
                summary = f"""
                {i}. {company.get('name', company.get('symbol', 'Unknown'))}
                   - SPR Score: {company.get('spr_score', 0):.2f}/10
                   - Sector: {company.get('sector', 'N/A')}
                   - Market Cap: ${company.get('market_cap', 0):,.0f}
                   - Sustainability Score: {company.get('sustainability_score', 0):.1f}/10
                   - Profit Performance: {company.get('profit_performance', 0):.1f}/10
                """
                company_summaries.append(summary)
            
            # Create the prompt
            prompt = f"""
            As an expert financial analyst specializing in sustainable investing, explain why these top 10 companies were selected based on the SPR (Sustainability Profit Ratio) analysis and NVIDIA cuOpt optimization.

            SELECTION CRITERIA:
            {json.dumps(selection_criteria, indent=2)}

            TOP 10 SELECTED COMPANIES:
            {''.join(company_summaries)}

            Please provide a comprehensive explanation covering:
            1. Why each company ranks in the top 10
            2. How the SPR scoring methodology favored these companies
            3. The role of NVIDIA cuOpt in the selection process
            4. Key sustainability and profitability factors
            5. Investment potential and risk considerations
            6. How these companies align with the search criteria
            
            Make the explanation engaging, informative, and suitable for both novice and experienced investors.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert financial analyst specializing in sustainable investing and ESG analysis. Provide detailed, accurate, and insightful explanations about investment selections."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.extend([
                {"role": "assistant", "content": explanation, "type": "explanation"}
            ])
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return f"I apologize, but I encountered an error while generating the explanation: {str(e)}"
    
    async def chat_about_companies(self, user_question: str, 
                                 companies_data: List[Dict[str, Any]]) -> str:
        """
        Handle user questions about the top 10 companies
        
        Args:
            user_question: User's question
            companies_data: Current top 10 companies data
            
        Returns:
            str: AI response to the question
        """
        if not self.client:
            return "AI chat not available - Mistral API key required for interactive chat. Please add your Mistral API key to the .env file."
        
        try:
            # Prepare context about companies
            company_context = self._prepare_company_context(companies_data)
            
            # Build conversation with context
            messages = [
                {"role": "system", "content": f"""
                You are an expert financial analyst and investment advisor with deep knowledge of sustainable investing. 
                You have access to detailed analysis of the top 10 companies selected through SPR analysis and NVIDIA cuOpt optimization.
                
                CURRENT TOP 10 COMPANIES CONTEXT:
                {company_context}
                
                Guidelines:
                - Provide accurate, helpful, and detailed responses
                - Reference specific data from the companies when relevant
                - Explain complex financial concepts in accessible terms
                - Focus on sustainability and profitability aspects
                - Suggest actionable insights when appropriate
                - Acknowledge limitations if you don't have specific information
                """}
            ]
            
            # Add conversation history (last 5 exchanges to maintain context)
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            for msg in recent_history:
                if msg.get("type") != "explanation":  # Don't include the initial explanation in chat context
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current user question
            messages.append({"role": "user", "content": user_question})
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": ai_response}
            ])
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error in chat response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
    
    def _prepare_company_context(self, companies_data: List[Dict[str, Any]]) -> str:
        """Prepare detailed context about companies for the AI"""
        context_parts = []
        
        for i, company in enumerate(companies_data[:10], 1):
            context = f"""
            Company #{i}: {company.get('name', company.get('symbol', 'Unknown'))}
            - Symbol: {company.get('symbol', 'N/A')}
            - SPR Score: {company.get('spr_score', 0):.2f}/10
            - Sector: {company.get('sector', 'N/A')}
            - Industry: {company.get('industry', 'N/A')}
            - Market Cap: ${company.get('market_cap', 0):,.0f}
            - Revenue: ${company.get('revenue', 0):,.0f}
            - Sustainability Score: {company.get('sustainability_score', 0):.1f}/10
            - Profit Performance: {company.get('profit_performance', 0):.1f}/10
            - ESG Score: {company.get('esg_score', 'N/A')}
            - Risk Level: {company.get('risk_level', 'N/A')}
            - Geographic Location: {company.get('location', 'N/A')}
            """
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    async def generate_portfolio_insights(self, companies_data: List[Dict[str, Any]], 
                                        analysis_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio insights using AI
        
        Args:
            companies_data: List of company data
            analysis_criteria: Analysis parameters
            
        Returns:
            Dict with insights and recommendations
        """
        if not self.client:
            return {
                "insights_text": "Portfolio insights not available - Mistral API key required.",
                "generated_at": datetime.now().isoformat(),
                "model_used": "demo_mode"
            }
        
        try:
            # Prepare portfolio summary
            portfolio_summary = []
            total_market_cap = sum(company.get('market_cap', 0) for company in companies_data)
            avg_spr_score = sum(company.get('spr_score', 0) for company in companies_data) / len(companies_data) if companies_data else 0
            
            sectors = {}
            for company in companies_data:
                sector = company.get('sector', 'Unknown')
                sectors[sector] = sectors.get(sector, 0) + 1
            
            portfolio_summary.append(f"Portfolio of {len(companies_data)} companies")
            portfolio_summary.append(f"Total Market Cap: ${total_market_cap:,.0f}")
            portfolio_summary.append(f"Average SPR Score: {avg_spr_score:.2f}/10")
            portfolio_summary.append(f"Sector Distribution: {sectors}")
            
            prompt = f"""
            Analyze this investment portfolio and provide comprehensive insights:
            
            PORTFOLIO SUMMARY:
            {chr(10).join(portfolio_summary)}
            
            ANALYSIS CRITERIA:
            {json.dumps(analysis_criteria, indent=2)}
            
            Provide insights on:
            1. Portfolio diversification and balance
            2. Risk assessment and management
            3. Sustainability profile and ESG considerations
            4. Growth potential and market outlook
            5. Recommended portfolio adjustments
            6. Key risks and opportunities
            
            Format your response as actionable investment insights.
            """
            
            messages = [
                {"role": "system", "content": "You are a senior portfolio manager specializing in sustainable investing. Provide structured, actionable investment insights."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.6
            )
            
            insights_text = response.choices[0].message.content
            
            return {
                "insights_text": insights_text,
                "generated_at": datetime.now().isoformat(),
                "model_used": self.model
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio insights: {e}")
            return {
                "insights_text": f"Error generating insights: {str(e)}",
                "generated_at": datetime.now().isoformat(),
                "model_used": "error"
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg["role"] == "assistant"]),
            "conversation_started": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_message": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
