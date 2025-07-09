"""
Enhanced PubMed Retriever for Healthcare Q&A Tool
Robust version with fallback mechanisms
"""

import time
from typing import Dict, List, Optional
from xml.etree import ElementTree
import requests
from loguru import logger
from tqdm import tqdm
from ..config import get_settings


class PubMedAPIError(Exception):
    """Custom exception for PubMed API errors"""
    pass


class PubMedRetriever:
    """PubMed article retriever with robust error handling"""
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self):
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{self.settings.pubmed_tool_name}/1.0'
        })
        self.last_request_time = 0
        self.rate_limit_delay = self.settings.pubmed_rate_limit_delay
        logger.info(f"Initialized PubMed retriever with rate limit: {self.rate_limit_delay}s")

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        if (wait_time := self.rate_limit_delay - (current_time - self.last_request_time)) > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict, retries: int = 3) -> requests.Response:
        """Make a robust HTTP request with retries"""
        self._rate_limit()
        
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    raise PubMedAPIError(f"API request failed after {retries} retries: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def search_articles(self, search_term: str, max_results: int = None, **kwargs) -> List[str]:
        """Search PubMed with multiple fallback strategies"""
        if max_results is None:
            max_results = self.settings.max_articles_per_search

        # Try different database parameters
        for db_option in ['pubmed', 'pmc', None]:  # None = no db parameter
            try:
                params = {
                    'term': search_term,
                    'retmax': min(100, max_results),
                    'retmode': 'xml',
                    'email': self.settings.pubmed_email or 'default@example.com',
                    'tool': self.settings.pubmed_tool_name or 'default_tool',
                    **({'db': db_option} if db_option else {})
                }
                
                response = self._make_request(self.SEARCH_URL, params)
                root = ElementTree.fromstring(response.content)
                
                if (error := root.find('.//ERROR')) is not None:
                    logger.warning(f"PubMed API error (db={db_option}): {error.text}")
                    continue
                    
                if (ids := [id_elem.text for id_elem in root.findall(".//Id")]):
                    return ids[:max_results]
                    
            except Exception as e:
                logger.error(f"Search failed (db={db_option}): {e}")
                continue

        # Final fallback to simplest possible search
        return self._direct_search(search_term, max_results)

    def _direct_search(self, search_term: str, max_results: int) -> List[str]:
        """Minimal search without most parameters"""
        try:
            response = requests.get(
                self.SEARCH_URL,
                params={'term': search_term, 'retmax': max_results},
                timeout=10
            )
            root = ElementTree.fromstring(response.content)
            return [id_elem.text for id_elem in root.findall(".//Id")]
        except Exception as e:
            logger.error(f"Direct search failed: {e}")
            return []

    def fetch_articles(self, pmid_list: List[str]) -> List[Dict]:
        """Fetch article details for given PMIDs"""
        if not pmid_list:
            return []

        articles = []
        batch_size = 100  # PubMed's limit per request
        
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i + batch_size]
            try:
                response = self._make_request(
                    self.FETCH_URL,
                    {'db': 'pubmed', 'id': ','.join(batch), 'retmode': 'xml'}
                )
                articles.extend(self._parse_articles(response.content))
            except Exception as e:
                logger.error(f"Failed to fetch batch {i//batch_size + 1}: {e}")

        return articles

    def _parse_articles(self, xml_content: bytes) -> List[Dict]:
        """Parse XML response into article dictionaries"""
        try:
            root = ElementTree.fromstring(xml_content)
            return [self._parse_article(article) 
                    for article in root.findall(".//PubmedArticle") 
                    if (parsed := self._parse_article(article))]
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
            return []

    def _parse_article(self, article_elem) -> Optional[Dict]:
        """Parse individual PubmedArticle element"""
        try:
            return {
                'pmid': self._safe_find_text(article_elem, ".//PMID"),
                'title': self._safe_find_text(article_elem, ".//ArticleTitle", "No Title"),
                'abstract': self._extract_abstract(article_elem),
                'journal': self._safe_find_text(article_elem, ".//Journal/Title", "Unknown Journal"),
                'authors': self._extract_authors(article_elem),
                'publication_date': self._extract_publication_date(article_elem),
                'doi': self._extract_doi(article_elem)
            }
        except Exception as e:
            pmid = self._safe_find_text(article_elem, ".//PMID", "unknown")
            logger.warning(f"Failed to parse article {pmid}: {e}")
            return None

    # Helper methods remain the same as before...
    def _safe_find_text(self, element, xpath: str, default: str = "") -> str:
        found = element.find(xpath)
        return found.text if found is not None and found.text else default

    def _extract_abstract(self, article_elem) -> Dict[str, str]:
        sections = article_elem.findall(".//AbstractText")
        return {section.attrib.get('Label', 'SUMMARY'): section.text
                for section in sections if section.text} or {"SUMMARY": "No Abstract"}

    def _extract_authors(self, article_elem) -> str:
        authors = []
        for author in article_elem.findall(".//Author"):
            if (last := self._safe_find_text(author, ".//LastName")):
                authors.append(last)
        return ", ".join(authors) if authors else "No Authors"

    def _extract_publication_date(self, article_elem) -> str:
        if (year := self._safe_find_text(article_elem, ".//PubDate/Year")):
            return "-".join(filter(None, [
                year,
                self._safe_find_text(article_elem, ".//PubDate/Month"),
                self._safe_find_text(article_elem, ".//PubDate/Day")
            ]))
        return self._safe_find_text(article_elem, ".//PubDate/MedlineDate", "Unknown Date")

    def _extract_doi(self, article_elem) -> Optional[str]:
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.attrib.get('IdType') == 'doi':
                return article_id.text
        return None

    def search_and_fetch(self, search_term: str, max_results: int = None, **kwargs) -> List[Dict]:
        """Convenience method to search and fetch in one call"""
        pmids = self.search_articles(search_term, max_results, **kwargs)
        return self.fetch_articles(pmids)