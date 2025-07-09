"""
ChromaDB Vector Store Manager for Healthcare Q&A Tool
Handles document storage, retrieval, and vector search operations.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from ..config import get_settings

logger = logging.getLogger(__name__)


class ChromaEmbeddingFunction(EmbeddingFunction):
    """Embedding function compatible with ChromaDB 0.4.16+ interface."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, convert_to_tensor=False).tolist()


class ChromaManager:
    """Manager for ChromaDB vector store operations."""

    def __init__(self, collection_name: Optional[str] = None):
        """Initialize the ChromaDB client and collection."""
        self.settings = get_settings()
        self.embedding_model = self._initialize_embedding_model()
        self.embedding_function = ChromaEmbeddingFunction(self.embedding_model)
        self.client = self._initialize_client()
        self.collection_name = collection_name or self.settings.chroma_collection_name
        self.collection = None
        logger.info(f"Initialized ChromaManager for collection: {self.collection_name}")

    def _initialize_embedding_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model for embeddings."""
        try:
            # Try loading model without specifying device first
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully without device specification")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model without device: {e}")
            try:
                # Fallback: load with device='cpu' explicitly
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                logger.info("Embedding model initialized successfully with device='cpu' fallback")
                return model
            except Exception as e2:
                logger.error(f"Failed to initialize embedding model with fallback: {e2}")
                raise

    def _initialize_client(self) -> chromadb.PersistentClient:
        """Initialize the ChromaDB client."""
        try:
            client = chromadb.PersistentClient(
                path=str(self.settings.chroma_path),
                settings=chromadb.Settings(allow_reset=True)
            )
            logger.info(f"Connected to Chroma DB at: {self.settings.chroma_path}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise

    def create_collection(self, reset_if_exists: bool = False) -> None:
        """
        Create a new collection or get existing one.

        Args:
            reset_if_exists: Whether to reset the collection if it already exists
        """
        try:
            if reset_if_exists:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass  # Collection might not exist

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Healthcare research articles from PubMed"}
            )

            logger.info(f"Collection '{self.collection_name}' ready")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(f"Collection creation failed: {e}")

    def get_collection(self) -> Collection:
        """Get the current collection, creating it if necessary."""
        if self.collection is None:
            self.create_collection()
        return self.collection

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the collection.

        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch

        Returns:
            Number of documents successfully added
        """
        collection = self.get_collection()
        added_count = 0

        logger.info(f"Adding {len(documents)} documents to collection")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            try:
                # Prepare batch data
                ids = []
                texts = []
                metadatas = []

                for doc in batch:
                    # Generate unique ID
                    doc_id = doc.get('pmid', None)
                    if doc_id is None:
                        import uuid
                        doc_id = str(uuid.uuid4())
                    ids.append(f"pmid_{doc_id}")

                    # Use full_text for embedding
                    text = doc.get('full_text', '')
                    if not text:
                        # Fallback to title + abstract
                        title = doc.get('title', '')
                        abstract = doc.get('abstract', {})
                        if isinstance(abstract, dict):
                            abstract_text = ' '.join(abstract.values())
                        else:
                            abstract_text = str(abstract)
                        text = f"{title}\n{abstract_text}"

                    texts.append(text)

                    # Prepare metadata (exclude large text fields)
                    metadata = {
                        'pmid': doc.get('pmid', ''),
                        'title': doc.get('title', ''),
                        'journal': doc.get('journal', ''),
                        'authors': doc.get('authors', ''),
                        'publication_date': doc.get('publication_date', ''),
                        'doi': doc.get('doi', ''),
                        'keywords': str(doc.get('keywords', [])),
                        'article_types': str(doc.get('article_types', [])),
                        'healthcare_relevance': float(doc.get('healthcare_relevance', 0.0)),
                        'research_focus': str(doc.get('research_focus', [])),
                        'study_type': doc.get('study_type', 'unknown')
                    }
                    metadatas.append(metadata)

                # Add to collection
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

                added_count += len(batch)
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} documents")

            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                continue

        logger.info(f"Successfully added {added_count} documents to collection")
        return added_count

    def search_documents(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the collection.

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with documents and metadata
        """
        collection = self.get_collection()

        try:
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'id': results['ids'][0][i],
                        'document': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} results for query: '{query[:50]}...'")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        collection = self.get_collection()

        try:
            count = collection.count()

            # Get sample of metadata to analyze
            sample_results = collection.peek(limit=min(100, count))

            # Analyze metadata
            journals = set()
            years = set()
            article_types = set()

            for metadata in sample_results.get('metadatas', []):
                if metadata.get('journal'):
                    journals.add(metadata['journal'])
                if metadata.get('publication_date'):
                    year = metadata['publication_date'].split('-')[0]
                    if year.isdigit():
                        years.add(year)
                if metadata.get('article_types'):
                    types = eval(metadata['article_types']) if metadata['article_types'] else []
                    article_types.update(types)

            stats = {
                'total_documents': count,
                'unique_journals': len(journals),
                'publication_years': sorted(list(years)),
                'article_types': list(article_types),
                'collection_name': self.collection_name
            }

            logger.info(f"Collection stats: {count} documents")
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}

    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents from the collection.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        collection = self.get_collection()

        try:
            collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return len(document_ids)

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Document deletion failed: {e}")

    def update_document(self, document_id: str, document: str, metadata: Dict) -> bool:
        """
        Update a single document in the collection.

        Args:
            document_id: ID of the document to update
            document: New document text
            metadata: New metadata

        Returns:
            True if successful
        """
        collection = self.get_collection()

        try:
            collection.update(
                ids=[document_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"Updated document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def reset_collection(self) -> bool:
        """Reset the collection by deleting and recreating it."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False

    def export_collection(self, output_path: str) -> bool:
        """
        Export collection data to a file.

        Args:
            output_path: Path to save the exported data

        Returns:
            True if successful
        """
        collection = self.get_collection()

        try:
            # Get all documents
            all_data = collection.get()

            import json
            export_data = {
                'collection_name': self.collection_name,
                'documents': all_data['documents'],
                'metadatas': all_data['metadatas'],
                'ids': all_data['ids']
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported collection to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            return False
