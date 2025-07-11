"""
Document processor for extracting and chunking text from PDF files.
Uses pdfplumber for robust PDF text extraction.
"""

import pdfplumber
import re
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF document processing and text chunking."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks (in words)
            overlap: Overlap between chunks (in words)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing headers, footers, and normalizing whitespace.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common footer patterns (page numbers, etc.)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with text content and metadata
        """
        try:
            document_text = ""
            page_texts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF: {pdf_path}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text from the page
                        page_text = page.extract_text()
                        
                        if page_text:
                            # Clean the text
                            cleaned_text = self.clean_text(page_text)
                            
                            if cleaned_text:
                                page_texts.append({
                                    'page_number': page_num,
                                    'text': cleaned_text
                                })
                                document_text += f"\n\n{cleaned_text}"
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
            
            # Clean the full document text
            document_text = self.clean_text(document_text)
            
            return {
                'text': document_text,
                'page_texts': page_texts,
                'source_file': Path(pdf_path).name,
                'total_pages': len(page_texts)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with NLTK if needed
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create text chunks from document data.
        
        Args:
            document_data: Document data from extract_text_from_pdf
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        text = document_data['text']
        source_file = document_data['source_file']
        
        # Split text into words
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Document is smaller than chunk size, return as single chunk
            chunks.append({
                'text': text,
                'source_file': source_file,
                'chunk_index': 0,
                'word_count': len(words),
                'page_range': f"1-{document_data['total_pages']}"
            })
            return chunks
        
        # Create overlapping chunks
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(words):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            # Extract chunk words
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Try to break at sentence boundaries if possible
            if end_idx < len(words):
                # Look for sentence endings in the last part of the chunk
                last_part = ' '.join(chunk_words[-50:])  # Last 50 words
                sentences = self.split_into_sentences(last_part)
                
                if len(sentences) > 1:
                    # Find the last complete sentence
                    last_sentence_end = chunk_text.rfind(sentences[-2])
                    if last_sentence_end > len(chunk_text) * 0.7:  # Only if we don't lose too much
                        chunk_text = chunk_text[:last_sentence_end + len(sentences[-2])]
            
            # Estimate page range for this chunk
            start_page = max(1, int((start_idx / len(words)) * document_data['total_pages']))
            end_page = min(document_data['total_pages'], 
                          int((end_idx / len(words)) * document_data['total_pages']) + 1)
            
            page_range = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
            
            chunks.append({
                'text': chunk_text.strip(),
                'source_file': source_file,
                'chunk_index': chunk_index,
                'word_count': len(chunk_text.split()),
                'page_range': page_range
            })
            
            chunk_index += 1
            
            # Calculate next start position with overlap
            start_idx = max(start_idx + self.chunk_size - self.overlap, start_idx + 1)
            
            # Break if we've processed all words
            if start_idx >= len(words):
                break
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF file and return chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks with metadata
        """
        try:
            # Extract text from PDF
            document_data = self.extract_text_from_pdf(pdf_path)
            
            # Create chunks
            chunks = self.create_chunks(document_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of all text chunks from all PDFs
        """
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return all_chunks
        
        # Find all PDF files
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return all_chunks
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Process a single PDF
    # chunks = processor.process_pdf("path/to/your/document.pdf")
    
    # Process all PDFs in a directory
    # chunks = processor.process_directory("data/")
    
    # Print chunk information
    # for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
    #     print(f"Chunk {i+1}:")
    #     print(f"  Source: {chunk['source_file']}")
    #     print(f"  Page Range: {chunk['page_range']}")
    #     print(f"  Word Count: {chunk['word_count']}")
    #     print(f"  Text Preview: {chunk['text'][:200]}...")
    #     print() 