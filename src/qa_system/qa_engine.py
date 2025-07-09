"""
Q&A Engine for Healthcare Q&A Tool.

This module provides intelligent question-answering capabilities
using retrieval-augmented generation (RAG) with healthcare literature.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import get_settings
from ..llm.openrouter_client import OpenRouterClient
from ..vector_store import ChromaManager


class QAEngineError(Exception):
    """Custom exception for Q&A engine errors."""
    pass


class QAEngine:
    """Intelligent Q&A engine with retrieval-augmented generation."""
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the Q&A engine.

        Args:
            collection_name: Name of the vector collection to use
        """
        self.settings = get_settings()
        self.chroma_manager = ChromaManager(collection_name)

        # Initialize Euri client
        self.llm_client = OpenRouterClient()

        logger.info("Initialized QAEngine with openrouter client")
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        n_results: int = 5,
        min_relevance_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question
            n_results: Number of documents to retrieve
            min_relevance_score: Minimum relevance score filter
            
        Returns:
            List of relevant documents
        """
        try:
            # First try without metadata filter to see if we get any results
            results = self.chroma_manager.search_documents(
                query=query,
                n_results=n_results * 2  # Get more results to filter manually
            )
            logger.debug(f"Retrieved {len(results)} documents from vector store for query '{query[:50]}...'")
            if results:
                         logger.debug(f"Retrieved {len(results)} documents from vector store for query '{query[:50]}...'")
                         if results:
                               for i, doc in enumerate(results, 1):
                                  doc_text = doc.get('content', '')
                                  metadata = doc.get('metadata', {})
                                  logger.debug(f"Document {i} snippet: {doc_text[:200]}")
                                  logger.debug(f"Document {i} metadata: {metadata}")
                                  logger.debug(f"Document {i} healthcare_relevance: {metadata.get('healthcare_relevance', 'N/A')}")
                                  logger.debug(f"Document {i} study_type: {metadata.get('study_type', 'N/A')}")
                                  logger.debug(f"Document {i} publication_date: {metadata.get('publication_date', 'N/A')}")
                                  logger.debug(f"Document {i} doi: {metadata.get('doi', 'N/A')}")

               # for i, doc in enumerate(results[:3], 1):
                  #  doc_text = doc.get('document', '')
                    #logger.debug(f"Document {i} snippet: {doc_text[:200]}")

            # If we have results, filter by relevance score manually
            if results and min_relevance_score > 0:
                filtered_results = []
                for result in results:
                    metadata = result.get('metadata', {})
                    relevance = float(metadata.get('healthcare_relevance', 0))
                    if relevance >= min_relevance_score:
                        filtered_results.append(result)

                # If manual filtering gives us results, use them
                if filtered_results:
                    results = filtered_results[:n_results]
                else:
                    # If no results pass the filter, lower the threshold and try again
                    logger.warning(f"No documents found with relevance >= {min_relevance_score}, using all results")
                    results = results[:n_results]

            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results

        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def generate_answer(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer using retrieved documents as context.
        
        Args:
            query: User question
            context_documents: Retrieved relevant documents
            include_sources: Whether to include source information
            
        Returns:
            Generated answer with metadata
        """
        try:
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            logger.debug(f"Context prepared for query '{query[:50]}...': {context[:500]}")

            # Generate response using OpenRouter client
            answer_text = self.llm_client.generate_healthcare_response(
                query=query,
                context=context
            )
            
            # Prepare response
            answer = {
                "question": query,
                "answer": answer_text,
                "confidence": self._calculate_confidence(context_documents),
                "model_used": self.settings.llm_model,
                "sources_count": len(context_documents)
            }
            
            if include_sources:
                answer["sources"] = self._format_sources(context_documents)
            
            logger.info(f"Generated answer for query: '{query[:50]}...'")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise QAEngineError(f"Failed to generate answer: {e}")
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            document_text = doc.get('content', '')
            
            # Create context entry
            context_entry = f"""
Document {i}:
Title: {metadata.get('title', 'Unknown')}
Authors: {metadata.get('authors', 'Unknown')}
Journal: {metadata.get('journal', 'Unknown')}
Publication Date: {metadata.get('publication_date', 'Unknown')}
Study Type: {metadata.get('study_type', 'Unknown')}

Content: {document_text[:1000]}...
"""
            context_parts.append(context_entry.strip())
        
        return "\n\n".join(context_parts)
    

    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on document quality and relevance."""
        if not documents:
            return 0.0
        
        total_score = 0.0
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            
            # Base score from healthcare relevance
            relevance = float(metadata.get('healthcare_relevance', 0))
            score = relevance * 0.4
            
            # Bonus for study type quality
            study_type = metadata.get('study_type', '')
            study_quality = {
                'randomized_controlled_trial': 0.3,
                'systematic_review': 0.25,
                'clinical_trial': 0.2,
                'cohort_study': 0.15,
                'case_control': 0.1,
                'cross_sectional': 0.05
            }
            score += study_quality.get(study_type, 0)
            
            # Bonus for recent publication
            pub_date = metadata.get('publication_date', '')
            if pub_date:
                year = pub_date.split('-')[0]
                if year.isdigit() and int(year) >= 2020:
                    score += 0.1
            
            # Bonus for having DOI
            if metadata.get('doi'):
                score += 0.05
            
            score = min(score, 1.0)
            total_score += score

            # Debug log for each document score
            import logging
            logging.debug(f"Doc {i} confidence score: {score} (relevance: {relevance}, study_type: {study_type}, pub_date: {pub_date}, doi: {metadata.get('doi')})")
        
        avg_score = min(total_score / len(documents), 1.0)
        logging.debug(f"Average confidence score: {avg_score}")
        return avg_score
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for the response."""
        sources = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            source = {
                'pmid': metadata.get('pmid', ''),
                'title': metadata.get('title', 'Unknown'),
                'authors': metadata.get('authors', 'Unknown'),
                'journal': metadata.get('journal', 'Unknown'),
                'publication_date': metadata.get('publication_date', 'Unknown'),
                'doi': metadata.get('doi', ''),
                'study_type': metadata.get('study_type', 'Unknown'),
                'relevance_score': metadata.get('healthcare_relevance', 0)
            }
            sources.append(source)
        
        return sources

    def ask_question(
        self,
        question: str,
        n_documents: int = 5,
        min_relevance: float = 0.0,  # Changed from 0.3 to 0.0 for better results
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete Q&A pipeline: retrieve documents and generate answer.

        Args:
            question: User question
            n_documents: Number of documents to retrieve
            min_relevance: Minimum relevance score for documents
            include_sources: Whether to include source information

        Returns:
            Complete Q&A response
        """
        logger.info(f"Processing question: '{question[:100]}...'")

        try:
            # Retrieve relevant documents
            documents = self.retrieve_relevant_documents(
                query=question,
                n_results=n_documents,
                min_relevance_score=min_relevance
            )

            if not documents:
                return {
                    "question": question,
                    "answer": "I couldn't find relevant research articles to answer your question. "
                             "Please try rephrasing your question or check if the topic is covered "
                             "in the available literature.",
                    "confidence": 0.0,
                    "sources_count": 0,
                    "sources": [],
                    "error": "No relevant documents found"
                }

            # Generate answer
            response = self.generate_answer(
                query=question,
                context_documents=documents,
                include_sources=include_sources
            )

            return response

        except Exception as e:
            logger.error(f"Q&A pipeline failed: {e}")
            return {
                "question": question,
                "answer": "I encountered an error while processing your question. "
                         "Please try again or contact support.",
                "confidence": 0.0,
                "sources_count": 0,
                "sources": [],
                "error": str(e)
            }

    def get_research_summary(
        self,
        topic: str,
        max_documents: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a research summary for a given topic.

        Args:
            topic: Research topic
            max_documents: Maximum documents to analyze

        Returns:
            Research summary
        """
        logger.info(f"Generating research summary for: '{topic}'")

        try:
            # Retrieve documents
            documents = self.retrieve_relevant_documents(
                query=topic,
                n_results=max_documents,
                min_relevance_score=0.4
            )

            if not documents:
                return {
                    "topic": topic,
                    "summary": f"No high-quality research articles found for '{topic}'.",
                    "document_count": 0,
                    "key_findings": [],
                    "study_types": {},
                    "publication_years": []
                }

            # Analyze documents
            analysis = self._analyze_documents(documents)

            # Generate summary using Euri client
            context = self._prepare_context(documents)
            summary_text = self.llm_client.generate_research_summary(
                topic=topic,
                context=context
            )

            return {
                "topic": topic,
                "summary": summary_text,
                "document_count": len(documents),
                "analysis": analysis,
                "confidence": self._calculate_confidence(documents)
            }

        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {
                "topic": topic,
                "summary": f"Error generating summary: {str(e)}",
                "document_count": 0,
                "error": str(e)
            }

    def _analyze_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a collection of documents for patterns and insights."""
        analysis = {
            "study_types": {},
            "publication_years": {},
            "journals": {},
            "research_focus_areas": {},
            "average_relevance": 0.0
        }

        total_relevance = 0.0

        for doc in documents:
            metadata = doc.get('metadata', {})

            # Study types
            study_type = metadata.get('study_type', 'unknown')
            analysis["study_types"][study_type] = analysis["study_types"].get(study_type, 0) + 1

            # Publication years
            pub_date = metadata.get('publication_date', '')
            if pub_date:
                year = pub_date.split('-')[0]
                if year.isdigit():
                    analysis["publication_years"][year] = analysis["publication_years"].get(year, 0) + 1

            # Journals
            journal = metadata.get('journal', 'unknown')
            analysis["journals"][journal] = analysis["journals"].get(journal, 0) + 1

            # Research focus areas
            focus_areas = metadata.get('research_focus', '')
            if focus_areas:
                try:
                    focus_list = eval(focus_areas) if isinstance(focus_areas, str) else focus_areas
                    for focus in focus_list:
                        analysis["research_focus_areas"][focus] = analysis["research_focus_areas"].get(focus, 0) + 1
                except:
                    pass

            # Relevance
            relevance = float(metadata.get('healthcare_relevance', 0))
            total_relevance += relevance

        if documents:
            analysis["average_relevance"] = total_relevance / len(documents)

        return analysis

    def get_collection_insights(self) -> Dict[str, Any]:
        """Get insights about the current document collection."""
        try:
            stats = self.chroma_manager.get_collection_stats()

            # Get sample documents for analysis
            sample_docs = self.chroma_manager.search_documents(
                query="healthcare research",
                n_results=50
            )

            if sample_docs:
                analysis = self._analyze_documents(sample_docs)
                stats["document_analysis"] = analysis

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection insights: {e}")
            return {"error": str(e)}





from src.llm.openrouter_client import OpenRouterClient

client = OpenRouterClient()
response = client.generate_healthcare_response("Hello, how are you?", "")
print(response)
