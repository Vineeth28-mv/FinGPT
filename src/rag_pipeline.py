"""
RAG Pipeline for Financial Q&A System.
Integrates document retrieval with language model generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline for answering questions based on financial documents."""
    
    def __init__(self, 
                 model_name: str = "google/flan-t5-base",
                 embedding_manager: Optional[EmbeddingManager] = None,
                 document_processor: Optional[DocumentProcessor] = None,
                 max_context_length: int = 1000,
                 n_retrieve: int = 5):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Name of the language model to use
            embedding_manager: EmbeddingManager instance
            document_processor: DocumentProcessor instance
            max_context_length: Maximum context length for the model
            n_retrieve: Number of documents to retrieve for context
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.n_retrieve = n_retrieve
        
        # Initialize components
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.document_processor = document_processor or DocumentProcessor()
        
        # Initialize the language model
        logger.info(f"Loading language model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"RAG Pipeline initialized with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def create_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the language model.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context with sources
        context_with_sources = ""
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"Source {i}: {chunk['metadata']['source_file']} (Pages {chunk['metadata']['page_range']})"
            context_with_sources += f"{source_info}\n{chunk['text']}\n\n"
        
        # Create the prompt
        prompt = f"""Based on the provided financial documents, answer the following question. 
If the answer cannot be found in the provided context, respond with "I cannot answer this question based on the provided documents."

Context from documents:
{context_with_sources}

Question: {query}

Answer based solely on the provided context:"""
        
        return prompt
    
    def retrieve_context(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User's question
            n_results: Number of results to retrieve (defaults to self.n_retrieve)
            
        Returns:
            List of relevant context chunks
        """
        n_results = n_results or self.n_retrieve
        
        try:
            # Search for similar documents
            search_results = self.embedding_manager.search_similar(
                query=query,
                n_results=n_results,
                include_distances=True
            )
            
            # Filter results based on similarity threshold
            relevant_chunks = []
            for result in search_results['results']:
                # Only include chunks with similarity score > 0.3 (adjust as needed)
                if result['similarity_score'] and result['similarity_score'] > 0.3:
                    relevant_chunks.append(result)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def truncate_context(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Truncate context to fit within model limits.
        
        Args:
            context_chunks: List of context chunks
            
        Returns:
            Truncated list of context chunks
        """
        truncated_chunks = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_length = len(chunk['text'].split())
            if total_length + chunk_length <= self.max_context_length:
                truncated_chunks.append(chunk)
                total_length += chunk_length
            else:
                # Try to fit a partial chunk
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # Only if we have reasonable space left
                    words = chunk['text'].split()
                    truncated_text = ' '.join(words[:remaining_length])
                    truncated_chunk = chunk.copy()
                    truncated_chunk['text'] = truncated_text
                    truncated_chunks.append(truncated_chunk)
                break
        
        logger.info(f"Truncated context to {len(truncated_chunks)} chunks ({total_length} words)")
        return truncated_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the language model.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        try:
            # Create prompt
            prompt = self.create_prompt(query, context_chunks)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the answer
            answer = answer.strip()
            
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def answer_question(self, query: str, n_retrieve: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query: User's question
            n_retrieve: Number of documents to retrieve (optional)
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing question: {query}")
            
            # Retrieve relevant context
            context_chunks = self.retrieve_context(query, n_retrieve)
            
            if not context_chunks:
                return {
                    'query': query,
                    'answer': "I cannot answer this question based on the provided documents.",
                    'sources': [],
                    'confidence': 0.0,
                    'total_sources': 0
                }
            
            # Truncate context if needed
            context_chunks = self.truncate_context(context_chunks)
            
            # Generate answer
            answer = self.generate_answer(query, context_chunks)
            
            # Prepare source information
            sources = []
            for chunk in context_chunks:
                sources.append({
                    'source_file': chunk['metadata']['source_file'],
                    'page_range': chunk['metadata']['page_range'],
                    'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'similarity_score': chunk['similarity_score']
                })
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(s['similarity_score'] for s in sources) / len(sources)
            confidence = min(avg_similarity, 1.0)
            
            result = {
                'query': query,
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'total_sources': len(sources)
            }
            
            logger.info(f"Question answered successfully with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'query': query,
                'answer': "I encountered an error while processing your question. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'total_sources': 0
            }
    
    def process_documents(self, pdf_directory: str) -> bool:
        """
        Process PDF documents and add them to the vector store.
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing documents from: {pdf_directory}")
            
            # Process PDFs
            chunks = self.document_processor.process_directory(pdf_directory)
            
            if not chunks:
                logger.warning("No chunks created from PDFs")
                return False
            
            # Add to vector store
            self.embedding_manager.add_documents(chunks)
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Get embedding manager stats
            embedding_stats = self.embedding_manager.get_collection_stats()
            
            # Add model information
            stats = {
                'model_name': self.model_name,
                'device': str(self.device),
                'max_context_length': self.max_context_length,
                'n_retrieve': self.n_retrieve,
                'embedding_stats': embedding_stats
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.embedding_manager.clear_collection()
            logger.info("All documents cleared from vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False


# Utility functions
def create_rag_pipeline(model_name: str = "google/flan-t5-base",
                       embedding_model: str = "all-MiniLM-L6-v2",
                       vector_store_path: str = "vector_store") -> RAGPipeline:
    """
    Create a RAG pipeline instance.
    
    Args:
        model_name: Language model name
        embedding_model: Embedding model name
        vector_store_path: Path to vector store
        
    Returns:
        RAGPipeline instance
    """
    embedding_manager = EmbeddingManager(
        model_name=embedding_model,
        persist_directory=vector_store_path
    )
    
    document_processor = DocumentProcessor()
    
    return RAGPipeline(
        model_name=model_name,
        embedding_manager=embedding_manager,
        document_processor=document_processor
    )


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Process documents (uncomment to use)
    # success = rag.process_documents("data/")
    
    # Answer a question (uncomment to use)
    # result = rag.answer_question("What was the revenue growth last quarter?")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {len(result['sources'])}")
    
    # Get system stats
    # stats = rag.get_system_stats()
    # print(f"System stats: {stats}")
    
    print("RAG Pipeline initialized successfully!") 