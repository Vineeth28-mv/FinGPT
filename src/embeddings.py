"""
Embeddings module for generating embeddings and managing vector store.
Uses SentenceTransformer for embeddings and ChromaDB for vector storage.
"""

import chromadb
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import os
import ssl
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector store operations."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "financial_documents",
                 persist_directory: str = "vector_store"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the SentenceTransformer model
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize the sentence transformer model with robust error handling
        self.model = self._load_model_robust(model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def _disable_ssl_warnings(self):
        """Disable SSL warnings and verification."""
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Disable SSL verification
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Set environment variables
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            logger.info("SSL verification disabled")
        except Exception as e:
            logger.warning(f"Could not disable SSL verification: {e}")
    
    def _load_model_robust(self, model_name: str):
        """Load model with multiple fallback strategies."""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        
        # Strategy 1: Try normal loading
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Normal loading failed: {e}")
        
        # Strategy 2: Disable SSL and try again
        try:
            logger.info("Attempting to load model with SSL verification disabled...")
            self._disable_ssl_warnings()
            
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model with SSL disabled: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"SSL-disabled loading failed: {e}")
        
        # Strategy 3: Try alternative models
        alternative_models = [
            "paraphrase-MiniLM-L6-v2",
            "msmarco-MiniLM-L6-cos-v5",
            "multi-qa-MiniLM-L6-cos-v1"
        ]
        
        for alt_model in alternative_models:
            try:
                logger.info(f"Trying alternative model: {alt_model}")
                model = SentenceTransformer(alt_model)
                logger.info(f"Successfully loaded alternative model: {alt_model}")
                self.model_name = alt_model  # Update the model name
                return model
            except Exception as e:
                logger.warning(f"Alternative model {alt_model} failed: {e}")
                continue
        
        # Strategy 4: Use a simple TF-IDF based embedding as final fallback
        logger.warning("All SentenceTransformer models failed. Using TF-IDF fallback.")
        return self._create_tfidf_fallback()
    
    def _create_tfidf_fallback(self):
        """Create a simple TF-IDF based embedding system as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        class TFIDFEmbedder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(
                    max_features=384,  # Match typical sentence transformer dimensions
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.is_fitted = False
                self.dimension = 384
            
            def encode(self, texts, show_progress_bar=False):
                if isinstance(texts, str):
                    texts = [texts]
                
                if not self.is_fitted:
                    # Fit on the texts
                    embeddings = self.vectorizer.fit_transform(texts)
                    self.is_fitted = True
                    logger.info("TF-IDF embedder fitted on input texts")
                else:
                    embeddings = self.vectorizer.transform(texts)
                
                # Convert to dense array and ensure consistent dimensions
                dense_embeddings = embeddings.toarray()
                
                # Pad or truncate to match expected dimensions
                if dense_embeddings.shape[1] < self.dimension:
                    padding = np.zeros((dense_embeddings.shape[0], 
                                      self.dimension - dense_embeddings.shape[1]))
                    dense_embeddings = np.hstack([dense_embeddings, padding])
                elif dense_embeddings.shape[1] > self.dimension:
                    dense_embeddings = dense_embeddings[:, :self.dimension]
                
                return dense_embeddings
        
        logger.info("Created TF-IDF fallback embedder")
        return TFIDFEmbedder()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Ensure embeddings are 2D array
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        try:
            # Extract texts and prepare data
            texts = [chunk['text'] for chunk in chunks]
            ids = [f"{chunk['source_file']}_chunk_{chunk['chunk_index']}" for chunk in chunks]
            
            # Prepare metadata (ChromaDB requires string values)
            metadatas = []
            for chunk in chunks:
                metadata = {
                    'source_file': chunk['source_file'],
                    'chunk_index': str(chunk['chunk_index']),
                    'word_count': str(chunk['word_count']),
                    'page_range': chunk['page_range']
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add to ChromaDB collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      n_results: int = 5,
                      include_distances: bool = True) -> Dict[str, Any]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            include_distances: Whether to include similarity distances
            
        Returns:
            Dictionary with search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([query])
            
            # Ensure query embedding is 2D
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if include_distances else None,
                    'similarity_score': 1 - results['distances'][0][i] if include_distances else None
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            
            return {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return {'query': query, 'results': [], 'total_results': 0}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(100, count)
            if count > 0:
                sample = self.collection.get(limit=sample_size)
                
                # Analyze source files
                source_files = set()
                for metadata in sample['metadatas']:
                    source_files.add(metadata['source_file'])
                
                stats = {
                    'total_chunks': count,
                    'unique_source_files': len(source_files),
                    'source_files': list(source_files),
                    'sample_size': sample_size,
                    'model_name': self.model_name,
                    'collection_name': self.collection_name
                }
            else:
                stats = {
                    'total_chunks': 0,
                    'unique_source_files': 0,
                    'source_files': [],
                    'sample_size': 0,
                    'model_name': self.model_name,
                    'collection_name': self.collection_name
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Cleared collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def delete_documents_by_source(self, source_file: str) -> None:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_file: Name of the source file to delete
        """
        try:
            # Get all documents from the source file
            results = self.collection.get(
                where={"source_file": source_file},
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                # Delete the documents
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from {source_file}")
            else:
                logger.info(f"No documents found for source file: {source_file}")
                
        except Exception as e:
            logger.error(f"Error deleting documents from {source_file}: {e}")
            raise
    
    def update_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Update documents in the vector store (delete old, add new).
        
        Args:
            chunks: List of document chunks to update
        """
        if not chunks:
            logger.warning("No chunks provided to update")
            return
        
        try:
            # Get unique source files from chunks
            source_files = set(chunk['source_file'] for chunk in chunks)
            
            # Delete existing documents for these source files
            for source_file in source_files:
                self.delete_documents_by_source(source_file)
            
            # Add new documents
            self.add_documents(chunks)
            
            logger.info(f"Updated documents for {len(source_files)} source files")
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise


# Utility functions
def create_embedding_manager(model_name: str = "all-MiniLM-L6-v2",
                           collection_name: str = "financial_documents",
                           persist_directory: str = "vector_store") -> EmbeddingManager:
    """
    Create an embedding manager instance.
    
    Args:
        model_name: Name of the SentenceTransformer model
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the vector store
        
    Returns:
        EmbeddingManager instance
    """
    return EmbeddingManager(
        model_name=model_name,
        collection_name=collection_name,
        persist_directory=persist_directory
    )


# Example usage
if __name__ == "__main__":
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Example chunks (normally would come from document_processor)
    example_chunks = [
        {
            'text': "The company's revenue increased by 15% year-over-year...",
            'source_file': "financial_report.pdf",
            'chunk_index': 0,
            'word_count': 150,
            'page_range': "1-2"
        },
        {
            'text': "Operating expenses decreased by 8% compared to last quarter...",
            'source_file': "financial_report.pdf", 
            'chunk_index': 1,
            'word_count': 120,
            'page_range': "2-3"
        }
    ]
    
    # Add documents to vector store
    # embedding_manager.add_documents(example_chunks)
    
    # Search for similar documents
    # results = embedding_manager.search_similar("revenue growth", n_results=3)
    
    # Get collection statistics
    # stats = embedding_manager.get_collection_stats()
    # print(f"Collection stats: {stats}")
    
    print("Embedding manager initialized successfully!") 