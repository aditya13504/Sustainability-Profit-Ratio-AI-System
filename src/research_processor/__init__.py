"""
Research Processor Module

This module handles academic research paper processing and analysis:
- Paper search from arXiv and Google Scholar
- NLP processing using transformers
- Sustainability insight extraction
- Paper deduplication and relevance scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper_analyzer import ResearchAnalyzer

__all__ = ['ResearchAnalyzer']
