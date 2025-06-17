"""
Text processing utilities for the SPR Analyzer
"""

import re
import string
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


class TextProcessor:
    """Utility class for text processing and analysis"""
    
    def __init__(self):
        """Initialize text processor with required NLTK data"""
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def _download_nltk_data(self):
        """Download required NLTK data if not present"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from text using TF-IDF-like approach
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords sorted by importance
        """
        if not text:
            return []
            
        # Clean and tokenize
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text)
        
        # Remove stop words and short words
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 3]
        
        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
        
    def extract_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract sentences that contain specific keywords
        
        Args:
            text: Text to search in
            keywords: List of keywords to search for
            
        Returns:
            List of sentences containing the keywords
        """
        if not text or not keywords:
            return []
            
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
                
        return relevant_sentences
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using simple word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        # Extract keywords from both texts
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def extract_numbers_and_percentages(self, text: str) -> Dict[str, List[float]]:
        """
        Extract numerical values and percentages from text
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            Dictionary with 'numbers' and 'percentages' lists
        """
        if not text:
            return {"numbers": [], "percentages": []}
            
        # Pattern for percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = [float(match) for match in re.findall(percentage_pattern, text)]
        
        # Pattern for numbers (including decimals)
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers = [float(match) for match in re.findall(number_pattern, text)
                  if float(match) not in percentages]  # Avoid duplicating percentages
        
        return {
            "numbers": numbers,
            "percentages": percentages
        }
        
    def get_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment analysis of text using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        if not text:
            return {"polarity": 0.0, "subjectivity": 0.0}
            
        blob = TextBlob(text)
        
        return {
            "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        }
        
    def extract_financial_terms(self, text: str) -> List[str]:
        """
        Extract financial terms from text
        
        Args:
            text: Text to search in
            
        Returns:
            List of financial terms found
        """
        financial_terms = [
            'profit', 'revenue', 'roi', 'return on investment', 'ebitda',
            'margin', 'earnings', 'cost', 'expense', 'investment',
            'valuation', 'growth', 'performance', 'efficiency',
            'savings', 'reduction', 'increase', 'decrease'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in financial_terms:
            if term in text_lower:
                found_terms.append(term)
                
        return found_terms
        
    def extract_sustainability_terms(self, text: str) -> List[str]:
        """
        Extract sustainability terms from text
        
        Args:
            text: Text to search in
            
        Returns:
            List of sustainability terms found
        """
        sustainability_terms = [
            'sustainability', 'sustainable', 'green', 'renewable',
            'carbon', 'emission', 'energy efficiency', 'recycling',
            'waste reduction', 'environmental', 'esg', 'clean energy',
            'solar', 'wind', 'biodiversity', 'circular economy'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in sustainability_terms:
            if term in text_lower:
                found_terms.append(term)
                
        return found_terms
        
    def summarize_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text based on sentence importance
        
        Args:
            text: Text to summarize
            max_points: Maximum number of key points to return
            
        Returns:
            List of key points
        """
        if not text:
            return []
            
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_points:
            return sentences
            
        # Score sentences based on keyword frequency and position
        sentence_scores = []
        keywords = self.extract_keywords(text, 10)
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (earlier sentences get higher score)
            position_score = (len(sentences) - i) / len(sentences)
            score += position_score * 0.3
            
            # Keyword score
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            keyword_score = keyword_count / len(keywords) if keywords else 0
            score += keyword_score * 0.7
            
            sentence_scores.append((sentence, score))
            
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sentence_scores[:max_points]]
