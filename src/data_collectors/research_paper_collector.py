import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
import arxiv
import scholarly
from datetime import datetime, timedelta
import json
import logging
import re
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET

class ResearchPaperCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive search terms for maximum research coverage
        self.search_terms = [
            # Core ESG-Finance relationships
            "ESG financial performance",
            "sustainability profitability correlation",
            "environmental social governance returns",
            "corporate sustainability financial impact",
            "ESG integration investment returns",
            "sustainable investing performance",
            
            # Environmental finance
            "green finance performance",
            "renewable energy profitability",
            "carbon pricing financial impact",
            "climate risk financial performance",
            "environmental regulations stock returns",
            "clean energy investment returns",
            
            # Social factors and finance
            "employee satisfaction stock performance",
            "diversity financial performance",
            "corporate social responsibility returns",
            "stakeholder capitalism profitability",
            "social impact investing returns",
            "human capital financial performance",
            
            # Governance and performance
            "corporate governance stock returns",
            "board diversity financial performance",
            "executive compensation performance",
            "transparency financial results",
            "shareholder value governance",
            "risk management financial impact",
            
            # Sector-specific research
            "sustainable agriculture profitability",
            "circular economy business value",
            "green technology investment returns",
            "sustainable supply chain profitability",
            "ESG mining industry performance",
            "renewable energy sector analysis",
            
            # Risk and return analysis
            "ESG risk factors portfolio",
            "sustainability risk premium",
            "climate transition risk finance",
            "stranded assets financial impact",
            "ESG factor investing returns",
            "sustainable finance risk models",
            
            # Regional and market studies
            "ESG performance emerging markets",
            "sustainable investing developed markets",
            "European ESG regulations impact",
            "Asian sustainability investing",
            "ESG disclosure financial impact",
            "sustainable finance policy effects"
        ]
        
        # Additional specialized terms
        self.specialized_terms = [
            "materiality assessment financial",
            "ESG scoring methodology",
            "sustainability reporting standards",
            "TCFD recommendations implementation",
            "SASB standards financial performance",
            "UN SDGs business impact",
            "Paris Agreement corporate compliance",
            "green bonds performance analysis",
            "sustainability-linked loans",
            "ESG derivatives market"
        ]
        
        self.all_search_terms = self.search_terms + self.specialized_terms
        
    async def collect_comprehensive_research(self) -> List[Dict[str, Any]]:
        """Collect research papers from multiple academic sources"""
        self.logger.info(f"Starting comprehensive research collection with {len(self.all_search_terms)} search terms...")
        
        all_papers = []
        
        # Collect from arXiv (computer science, economics, finance)
        self.logger.info("Collecting from arXiv...")
        arxiv_papers = await self._collect_arxiv_papers()
        all_papers.extend(arxiv_papers)
        self.logger.info(f"Collected {len(arxiv_papers)} papers from arXiv")
        
        # Collect from Google Scholar (broad academic coverage)
        self.logger.info("Collecting from Google Scholar...")
        scholar_papers = await self._collect_scholar_papers()
        all_papers.extend(scholar_papers)
        self.logger.info(f"Collected {len(scholar_papers)} papers from Google Scholar")
        
        # Collect from SSRN (Social Science Research Network)
        self.logger.info("Collecting from SSRN...")
        ssrn_papers = await self._collect_ssrn_papers()
        all_papers.extend(ssrn_papers)
        self.logger.info(f"Collected {len(ssrn_papers)} papers from SSRN")
        
        # Collect from PubMed (for health and environmental research)
        self.logger.info("Collecting from PubMed...")
        pubmed_papers = await self._collect_pubmed_papers()
        all_papers.extend(pubmed_papers)
        self.logger.info(f"Collected {len(pubmed_papers)} papers from PubMed")
        
        # Collect from specific sustainability journals
        self.logger.info("Collecting from sustainability journals...")
        journal_papers = await self._collect_journal_papers()
        all_papers.extend(journal_papers)
        self.logger.info(f"Collected {len(journal_papers)} papers from journals")
        
        # Remove duplicates and enhance metadata
        unique_papers = self._remove_duplicate_papers(all_papers)
        enhanced_papers = self._enhance_paper_metadata(unique_papers)
        
        self.logger.info(f"Total unique papers collected: {len(enhanced_papers)}")
        return enhanced_papers
    
    async def _collect_arxiv_papers(self) -> List[Dict[str, Any]]:
        """Collect papers from arXiv"""
        papers = []
        max_results_per_search = 50
        
        for i, search_term in enumerate(self.all_search_terms):
            try:
                self.logger.debug(f"ArXiv search {i+1}/{len(self.all_search_terms)}: {search_term}")
                
                # Create search query
                search = arxiv.Search(
                    query=search_term,
                    max_results=max_results_per_search,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                # Collect results
                search_papers = []
                for result in search.results():
                    paper_data = {
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary,
                        'published_date': result.published.isoformat(),
                        'updated_date': result.updated.isoformat() if result.updated else None,
                        'url': result.entry_id,
                        'pdf_url': result.pdf_url,
                        'doi': result.doi,
                        'journal_ref': result.journal_ref,
                        'categories': result.categories,
                        'source': 'arxiv',
                        'search_term': search_term,
                        'collection_timestamp': datetime.now().isoformat(),
                        'relevance_score': self._calculate_relevance_score(result.title, result.summary, search_term)
                    }
                    search_papers.append(paper_data)
                
                papers.extend(search_papers)
                self.logger.debug(f"Found {len(search_papers)} papers for '{search_term}'")
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"ArXiv search error for '{search_term}': {e}")
                continue
                
        return papers
    
    async def _collect_scholar_papers(self) -> List[Dict[str, Any]]:
        """Collect papers from Google Scholar"""
        papers = []
        max_results_per_search = 20  # Lower limit to avoid blocking
        
        # Use only high-priority search terms for Scholar to avoid rate limiting
        priority_terms = self.search_terms[:15]  
        
        for i, search_term in enumerate(priority_terms):
            try:
                self.logger.debug(f"Scholar search {i+1}/{len(priority_terms)}: {search_term}")
                
                search_query = scholarly.search_pubs(search_term)
                
                search_papers = []
                for j, pub in enumerate(search_query):
                    if j >= max_results_per_search:
                        break
                    
                    # Get detailed publication info
                    try:
                        pub_filled = scholarly.fill(pub)
                    except:
                        pub_filled = pub
                    
                    paper_data = {
                        'title': pub_filled.get('title', ''),
                        'authors': pub_filled.get('author', []),
                        'abstract': pub_filled.get('abstract', ''),
                        'published_date': str(pub_filled.get('pub_year', '')),
                        'url': pub_filled.get('pub_url', ''),
                        'venue': pub_filled.get('venue', ''),
                        'citations': pub_filled.get('num_citations', 0),
                        'source': 'google_scholar',
                        'search_term': search_term,
                        'collection_timestamp': datetime.now().isoformat(),
                        'relevance_score': self._calculate_relevance_score(
                            pub_filled.get('title', ''), 
                            pub_filled.get('abstract', ''), 
                            search_term
                        )
                    }
                    search_papers.append(paper_data)
                
                papers.extend(search_papers)
                self.logger.debug(f"Found {len(search_papers)} papers for '{search_term}'")
                
                # Longer rate limiting for Scholar
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Scholar search error for '{search_term}': {e}")
                continue
                
        return papers
    
    async def _collect_ssrn_papers(self) -> List[Dict[str, Any]]:
        """Collect papers from SSRN (Social Science Research Network)"""
        papers = []
        
        # SSRN search would require web scraping or API access
        # For now, we'll implement a placeholder that could be expanded
        try:
            self.logger.debug("SSRN collection placeholder - would implement web scraping")
            # This would involve scraping SSRN search results
            # or using their API if available
            
            # Placeholder structure for SSRN papers
            papers = []
            
        except Exception as e:
            self.logger.error(f"SSRN collection error: {e}")
            
        return papers
    
    async def _collect_pubmed_papers(self) -> List[Dict[str, Any]]:
        """Collect papers from PubMed"""
        papers = []
        
        # Environmental and health-related search terms
        pubmed_terms = [
            "environmental health economics",
            "climate change health costs",
            "air pollution economic impact",
            "water quality economic assessment",
            "occupational health economics",
            "sustainable agriculture health"
        ]
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        for search_term in pubmed_terms:
            try:
                # Search PubMed
                search_url = f"{base_url}esearch.fcgi"
                search_params = {
                    'db': 'pubmed',
                    'term': search_term,
                    'retmax': 50,
                    'retmode': 'xml'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=search_params) as response:
                        if response.status == 200:
                            search_xml = await response.text()
                            
                            # Parse search results
                            root = ET.fromstring(search_xml)
                            id_list = [id_elem.text for id_elem in root.findall('.//Id')]
                            
                            if id_list:
                                # Fetch details for found papers
                                fetch_url = f"{base_url}efetch.fcgi"
                                fetch_params = {
                                    'db': 'pubmed',
                                    'id': ','.join(id_list[:20]),  # Limit to 20 papers
                                    'retmode': 'xml'
                                }
                                
                                async with session.get(fetch_url, params=fetch_params) as fetch_response:
                                    if fetch_response.status == 200:
                                        fetch_xml = await fetch_response.text()
                                        pubmed_papers = self._parse_pubmed_xml(fetch_xml, search_term)
                                        papers.extend(pubmed_papers)
                
                await asyncio.sleep(1)  # Rate limiting for PubMed
                
            except Exception as e:
                self.logger.error(f"PubMed search error for '{search_term}': {e}")
                continue
                
        return papers
    
    def _parse_pubmed_xml(self, xml_content: str, search_term: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract article details
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ''
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ''
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")
                    
                    # Extract publication date
                    pub_date = article.find('.//PubDate')
                    year = pub_date.find('Year').text if pub_date is not None and pub_date.find('Year') is not None else ''
                    
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''
                    
                    paper_data = {
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'published_date': year,
                        'pmid': pmid,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '',
                        'source': 'pubmed',
                        'search_term': search_term,
                        'collection_timestamp': datetime.now().isoformat(),
                        'relevance_score': self._calculate_relevance_score(title, abstract, search_term)
                    }
                    
                    papers.append(paper_data)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing PubMed article: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error parsing PubMed XML: {e}")
            
        return papers
    
    async def _collect_journal_papers(self) -> List[Dict[str, Any]]:
        """Collect papers from specific sustainability and finance journals"""
        papers = []
        
        # This would involve scraping specific journal websites
        # or using their APIs if available
        
        target_journals = [
            'Journal of Sustainable Finance & Investment',
            'Corporate Social Responsibility and Environmental Management',
            'Business Strategy and the Environment',
            'Journal of Business Ethics',
            'Sustainability Accounting, Management and Policy Journal',
            'Journal of Cleaner Production'
        ]
        
        try:
            self.logger.debug("Journal collection placeholder - would implement journal-specific scraping")
            # This would involve implementing specific scrapers for each journal
            # or using academic APIs like Crossref
            
        except Exception as e:
            self.logger.error(f"Journal collection error: {e}")
            
        return papers
    
    def _calculate_relevance_score(self, title: str, abstract: str, search_term: str) -> float:
        """Calculate relevance score for a paper based on search term"""
        try:
            text = f"{title} {abstract}".lower()
            search_words = search_term.lower().split()
            
            score = 0
            total_words = len(search_words)
            
            for word in search_words:
                if word in text:
                    score += 1
                    
            # Boost score for title matches
            title_lower = title.lower()
            for word in search_words:
                if word in title_lower:
                    score += 0.5
                    
            return score / total_words if total_words > 0 else 0
            
        except:
            return 0
    
    def _remove_duplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title and DOI"""
        seen_titles = set()
        seen_dois = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            doi = paper.get('doi', '').strip()
            
            # Skip if we've seen this title or DOI before
            is_duplicate = False
            
            if title and title in seen_titles:
                is_duplicate = True
            elif doi and doi in seen_dois:
                is_duplicate = True
            
            if not is_duplicate:
                unique_papers.append(paper)
                if title:
                    seen_titles.add(title)
                if doi:
                    seen_dois.add(doi)
        
        self.logger.info(f"Removed {len(papers) - len(unique_papers)} duplicate papers")
        return unique_papers
    
    def _enhance_paper_metadata(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance paper metadata with additional analysis"""
        enhanced_papers = []
        
        for paper in papers:
            try:
                # Add research categories
                paper['research_categories'] = self._categorize_research(paper)
                
                # Add quality indicators
                paper['quality_indicators'] = self._assess_paper_quality(paper)
                
                # Add ESG focus areas
                paper['esg_focus_areas'] = self._identify_esg_focus(paper)
                
                # Add methodological approach
                paper['methodology'] = self._identify_methodology(paper)
                
                enhanced_papers.append(paper)
                
            except Exception as e:
                self.logger.error(f"Error enhancing paper metadata: {e}")
                enhanced_papers.append(paper)  # Add original if enhancement fails
                
        return enhanced_papers
    
    def _categorize_research(self, paper: Dict[str, Any]) -> List[str]:
        """Categorize research paper by type"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = f"{title} {abstract}"
        
        categories = []
        
        # Research type categories
        if any(keyword in text for keyword in ['empirical', 'data', 'analysis', 'study', 'evidence']):
            categories.append('empirical')
        if any(keyword in text for keyword in ['theory', 'theoretical', 'model', 'framework']):
            categories.append('theoretical')
        if any(keyword in text for keyword in ['review', 'meta-analysis', 'survey', 'literature']):
            categories.append('review')
        if any(keyword in text for keyword in ['case study', 'case', 'company', 'firm']):
            categories.append('case_study')
        
        # Subject categories
        if any(keyword in text for keyword in ['environmental', 'climate', 'carbon', 'green']):
            categories.append('environmental')
        if any(keyword in text for keyword in ['social', 'diversity', 'labor', 'employee']):
            categories.append('social')
        if any(keyword in text for keyword in ['governance', 'board', 'management', 'transparency']):
            categories.append('governance')
        if any(keyword in text for keyword in ['financial', 'performance', 'returns', 'profitability']):
            categories.append('financial')
        
        return categories
    
    def _assess_paper_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Assess paper quality indicators"""
        quality = {
            'has_abstract': bool(paper.get('abstract')),
            'has_doi': bool(paper.get('doi')),
            'citation_count': paper.get('citations', 0),
            'author_count': len(paper.get('authors', [])),
            'title_length': len(paper.get('title', '')),
            'abstract_length': len(paper.get('abstract', '')),
            'venue_quality': 'unknown'
        }
        
        # Assess venue quality based on journal/venue name
        venue = paper.get('venue', '').lower()
        high_quality_venues = [
            'journal of finance', 'review of financial studies', 'journal of financial economics',
            'nature', 'science', 'proceedings of the national academy',
            'journal of business ethics', 'business strategy and the environment'
        ]
        
        if any(hq_venue in venue for hq_venue in high_quality_venues):
            quality['venue_quality'] = 'high'
        elif 'journal' in venue or 'review' in venue:
            quality['venue_quality'] = 'medium'
        
        return quality
    
    def _identify_esg_focus(self, paper: Dict[str, Any]) -> Dict[str, bool]:
        """Identify ESG focus areas in the paper"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = f"{title} {abstract}"
        
        focus_areas = {
            'environmental_focus': any(keyword in text for keyword in [
                'environmental', 'climate', 'carbon', 'emission', 'renewable', 'sustainability',
                'green', 'pollution', 'waste', 'water', 'energy'
            ]),
            'social_focus': any(keyword in text for keyword in [
                'social', 'diversity', 'employee', 'labor', 'human rights', 'community',
                'stakeholder', 'safety', 'health', 'education'
            ]),
            'governance_focus': any(keyword in text for keyword in [
                'governance', 'board', 'management', 'transparency', 'ethics', 'compliance',
                'risk', 'audit', 'shareholder', 'executive'
            ]),
            'financial_focus': any(keyword in text for keyword in [
                'financial', 'performance', 'returns', 'profitability', 'value', 'cost',
                'investment', 'market', 'stock', 'portfolio'
            ])
        }
        
        return focus_areas
    
    def _identify_methodology(self, paper: Dict[str, Any]) -> Dict[str, bool]:
        """Identify methodological approach used in the paper"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = f"{title} {abstract}"
        
        methodology = {
            'quantitative': any(keyword in text for keyword in [
                'regression', 'analysis', 'model', 'data', 'statistical', 'econometric',
                'panel', 'cross-section', 'time series', 'correlation'
            ]),
            'qualitative': any(keyword in text for keyword in [
                'interview', 'survey', 'case', 'qualitative', 'content analysis'
            ]),
            'mixed_methods': any(keyword in text for keyword in [
                'mixed method', 'triangulation', 'multi-method'
            ]),
            'experimental': any(keyword in text for keyword in [
                'experiment', 'treatment', 'control', 'randomized'
            ]),
            'simulation': any(keyword in text for keyword in [
                'simulation', 'monte carlo', 'bootstrap'
            ])
        }
        
        return methodology
