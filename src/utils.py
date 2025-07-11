"""
Utility functions for the Financial RAG System.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory_path}")


def get_pdf_files(directory_path: str) -> List[str]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        List of PDF file paths
    """
    directory = Path(directory_path)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory_path}")
        return []
    
    pdf_files = list(directory.glob("*.pdf"))
    pdf_paths = [str(f) for f in pdf_files]
    
    logger.info(f"Found {len(pdf_paths)} PDF files in {directory_path}")
    return pdf_paths


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    if not path.exists():
        return {'error': f"File does not exist: {file_path}"}
    
    stat = path.stat()
    
    return {
        'name': path.name,
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'extension': path.suffix.lower(),
        'is_file': path.is_file(),
        'is_dir': path.is_dir()
    }


def save_json(data: Any, file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\ufeff', '')  # Remove BOM
    
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if a file is a valid PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check if it's a file
        if not path.is_file():
            logger.error(f"Not a file: {file_path}")
            return False
        
        # Check file extension
        if path.suffix.lower() != '.pdf':
            logger.error(f"Not a PDF file: {file_path}")
            return False
        
        # Check file size (should be > 0)
        if path.stat().st_size == 0:
            logger.error(f"Empty file: {file_path}")
            return False
        
        # Basic PDF header check
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                logger.error(f"Invalid PDF header: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating PDF {file_path}: {e}")
        return False


def create_backup(file_path: str) -> Optional[str]:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file or None if error
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_{timestamp}_backup{path.suffix}"
        backup_path = path.parent / backup_name
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {e}")
        return None


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('.').percent,
            'timestamp': datetime.now().isoformat()
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources for display.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted sources string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        source_file = source.get('source_file', 'Unknown')
        page_range = source.get('page_range', 'Unknown')
        similarity = source.get('similarity_score', 0.0)
        
        formatted.append(f"{i}. {source_file} (Pages: {page_range}) - Similarity: {similarity:.2f}")
    
    return '\n'.join(formatted)


# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_MODEL_NAME = "google/flan-t5-base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_VECTOR_STORE_PATH = "vector_store"
DEFAULT_DATA_PATH = "data"

# File extensions
SUPPORTED_EXTENSIONS = ['.pdf']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
TEXT_EXTENSIONS = ['.txt', '.md', '.csv', '.json', '.xml'] 