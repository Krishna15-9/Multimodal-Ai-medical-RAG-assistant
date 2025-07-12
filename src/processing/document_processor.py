"""
Document Processing and Ingestion Pipeline for Healthcare Q&A Tool.

This module provides comprehensive document processing capabilities,
including text cleaning, chunking, and metadata enrichment.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from ..config import get_settings
from ..data_retrieval import PubMedRetriever
from ..vector_store import ChromaManager


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class DocumentProcessor:
    """Comprehensive document processor for PubMed articles."""

    def __init__(self):
        """Initialize the document processor."""
        self.settings = get_settings()
        self.pubmed_retriever = PubMedRetriever()
        self.chroma_manager = ChromaManager()
        logger.info("Initialized DocumentProcessor")

    def search_and_ingest_pipeline(
        self, 
        search_term: str, 
        max_results: int = 10, 
        reset_collection: bool = False,
        test_simple_queries: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete pipeline from search to ingestion."""
        try:
            logger.info(f"Starting complete pipeline for search: '{search_term}'")
            
            # Search PubMed with additional kwargs
            articles = self.pubmed_retriever.search_articles(search_term, max_results, **kwargs)
            
            # If no articles found and test_simple_queries is True, try simpler queries
            if not articles and test_simple_queries:
                logger.warning("No articles found for full query, trying simpler queries")
                simple_terms = search_term.split()
                for term in simple_terms:
                    logger.info(f"Trying simpler query: '{term}'")
                    articles = self.pubmed_retriever.search_articles(term, max_results, **kwargs)
                    if articles:
                        logger.info(f"Found articles for simpler query: '{term}'")
                        break
            
            if not articles:
                logger.warning("No articles found from PubMed search")
                return {
                    "success": False,
                    "error": "No articles found",
                    "articles_found_in_pubmed": 0
                }

            # Fetch article details
            fetched_articles = self.pubmed_retriever.fetch_articles(articles)
            if not fetched_articles:
                logger.warning("No articles successfully fetched")
                return {
                    "success": False,
                    "error": "No articles fetched", 
                    "articles_found_in_pubmed": len(articles)
                }

            # Ingest to vector store
            return self.ingest_articles_to_vector_store(fetched_articles, reset_collection)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def ingest_articles_to_vector_store(
        self, 
        articles: List[Dict[str, Any]], 
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """Process and ingest articles into vector store."""
        if not articles:
            logger.error("No articles provided for ingestion")
            return {
                "success": False,
                "error": "No articles provided",
                "total_articles_processed": 0,
                "articles_added_to_vector_store": 0,
            }

        logger.info(f"Starting ingestion pipeline for {len(articles)} articles")

        try:
            if reset_collection:
                self.chroma_manager.reset_collection()
                logger.info("Reset vector collection")

            # Process articles
            processed_articles = self.process_articles(articles)
            high_relevance_articles = [
                article for article in processed_articles 
                if article.get("healthcare_relevance", 0) > 0.3
            ]

            logger.info(
                f"Filtered to {len(high_relevance_articles)} high-relevance articles "
                f"from {len(processed_articles)} total"
            )

            # Prepare chunks and metadata
            chunks = []
            metadatas = []
            for article in high_relevance_articles:
                article_chunks = self.chunk_document(article)
                for chunk in article_chunks:
                    chunks.append(chunk["content"])
                    metadatas.append(chunk["metadata"])

            if not chunks:
                logger.error("No content to ingest")
                raise DocumentProcessingError("No content to ingest")

            # Combine chunks and metadatas into list of document dicts
            documents_to_add = []
            for content, metadata in zip(chunks, metadatas):
                doc = {
                    "full_text": content,
                    **metadata
                }
                documents_to_add.append(doc)

            self.chroma_manager.add_documents(documents_to_add)
            stats = self.chroma_manager.get_collection_stats()

            logger.info(f"Ingestion completed successfully: {stats}")
            return {
                "success": True,
                "total_articles_processed": len(processed_articles),
                "high_relevance_articles": len(high_relevance_articles),
                "articles_added_to_vector_store": len(chunks),
                "articles_found_in_pubmed": len(articles),
                "collection_stats": stats,
            }

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_articles_processed": len(articles),
                "articles_added_to_vector_store": 0,
            }

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process articles and calculate relevance scores."""
        logger.info(f"Processing {len(articles)} articles for relevance scoring")
        processed = []
        
        for article in tqdm(articles):
            if not article:  # Skip None articles
                continue
                
            try:
                content_parts = []
                
                # Handle title
                if article.get("title"):
                    content_parts.append(str(article["title"]))
                    
                # Handle abstract (convert dict to string if needed)
                abstract = article.get("abstract", {})
                if isinstance(abstract, dict):
                    abstract_text = " ".join(
                        [f"{k}: {v}" for k, v in abstract.items() if v]
                    )
                else:
                    abstract_text = str(abstract)
                    
                if abstract_text:
                    content_parts.append(abstract_text)

                combined_text = " ".join(content_parts).lower()

                # Calculate relevance score
                relevance = 0.0
                keywords = [
                    "health", "disease", "treatment", 
                    "obesity", "diabetes", "cancer", 
                    "headache", "medical", "clinical"
                ]
                if any(keyword in combined_text for keyword in keywords):
                    relevance = 0.7

                # Add more granular scoring based on study_type or publication_date if available
                study_type = article.get("study_type", "").lower()
                if study_type in ["randomized_controlled_trial", "systematic_review", "clinical_trial"]:
                    relevance += 0.1

                pub_date = article.get("publication_date", "")
                if pub_date:
                    year = pub_date.split("-")[0]
                    if year.isdigit() and int(year) >= 2020:
                        relevance += 0.05

                # Cap relevance at 1.0
                relevance = min(relevance, 1.0)

                # Create processed article copy
                processed_article = article.copy()
                processed_article["healthcare_relevance"] = relevance
                processed.append(processed_article)

            except Exception as e:
                logger.warning(f"Error processing article {article.get('pmid')}: {e}")
                continue

        logger.info(f"Completed processing of {len(processed)} articles")
        return processed

    def chunk_document(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks for vector storage."""
        chunks = []
        text_parts = []

        # Handle title
        if article.get("title"):
            text_parts.append(str(article["title"]))
            
        # Handle abstract
        abstract = article.get("abstract", {})
        if isinstance(abstract, dict):
            abstract_text = "\n".join(
                [f"{k}: {v}" for k, v in abstract.items() if v]
            )
        else:
            abstract_text = str(abstract)
            
        if abstract_text:
            text_parts.append(abstract_text)

        combined_text = " ".join(text_parts).strip()

        if not combined_text:
            logger.warning(f"Skipping article with no title or abstract: {article.get('pmid')}")
            return []

        # Prepare chunk metadata with sanitization
        def sanitize_metadata(value):
            if value is None:
                return ""
            if isinstance(value, dict):
                return str(value)
            return value

        year = ""
        if article.get("publication_date"):
            year = article["publication_date"].split("-")[0]

        chunk = {
            "content": combined_text,
            "metadata": {
                "pubmed_id": sanitize_metadata(article.get("pmid")),
                "title": sanitize_metadata(article.get("title")),
                "journal": sanitize_metadata(article.get("journal")),
                "year": year,
                "healthcare_relevance": sanitize_metadata(article.get("healthcare_relevance", 0)),
                "authors": sanitize_metadata(article.get("authors", "")),
                "article_types": ", ".join(article.get("article_types", [])),
                "study_type": sanitize_metadata(article.get("study_type", "unknown")),
                "publication_date": sanitize_metadata(article.get("publication_date", "")),
                "doi": sanitize_metadata(article.get("doi", "")),
            },
        }
        chunks.append(chunk)
        return chunks
