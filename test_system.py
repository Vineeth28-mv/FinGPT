#!/usr/bin/env python3
"""
Test script for Financial RAG System.
Verifies that all components are working correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("  ✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"  ❌ Streamlit import failed: {e}")
        return False
    
    try:
        import chromadb
        print("  ✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"  ❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import pdfplumber
        print("  ✅ PDFplumber imported successfully")
    except ImportError as e:
        print(f"  ❌ PDFplumber import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("  ✅ Transformers imported successfully")
    except ImportError as e:
        print(f"  ❌ Transformers import failed: {e}")
        return False
    
    return True

def test_embedding_manager():
    """Test the embedding manager."""
    print("\n🧪 Testing embedding manager...")
    
    try:
        from src.embeddings import EmbeddingManager
        
        # Create embedding manager (this will test model loading)
        embedding_manager = EmbeddingManager()
        print("  ✅ Embedding manager created successfully")
        
        # Test embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = embedding_manager.generate_embeddings(test_texts)
        print(f"  ✅ Generated embeddings shape: {embeddings.shape}")
        
        # Test adding documents
        test_chunks = [
            {
                'text': "Test financial document content.",
                'source_file': "test.pdf",
                'chunk_index': 0,
                'word_count': 5,
                'page_range': "1"
            }
        ]
        
        embedding_manager.add_documents(test_chunks)
        print("  ✅ Documents added to vector store")
        
        # Test search
        results = embedding_manager.search_similar("financial document", n_results=1)
        print(f"  ✅ Search completed, found {results['total_results']} results")
        
        # Clean up
        embedding_manager.clear_collection()
        print("  ✅ Vector store cleared")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Embedding manager test failed: {e}")
        return False

def test_document_processor():
    """Test the document processor."""
    print("\n🧪 Testing document processor...")
    
    try:
        from src.document_processor import DocumentProcessor
        
        # Create document processor
        processor = DocumentProcessor()
        print("  ✅ Document processor created successfully")
        
        # Test text cleaning
        test_text = "  This   is  a    test  text  with  extra   spaces  "
        cleaned = processor.clean_text(test_text)
        print(f"  ✅ Text cleaning works: '{cleaned}'")
        
        # Test chunking with fake document data
        fake_doc_data = {
            'text': "This is a test document. " * 100,  # 500 words
            'source_file': "test.pdf",
            'total_pages': 1
        }
        
        chunks = processor.create_chunks(fake_doc_data)
        print(f"  ✅ Created {len(chunks)} chunks from test document")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Document processor test failed: {e}")
        return False

def test_rag_pipeline():
    """Test the RAG pipeline (basic initialization)."""
    print("\n🧪 Testing RAG pipeline...")
    
    try:
        from src.rag_pipeline import create_rag_pipeline
        
        # Create RAG pipeline
        rag = create_rag_pipeline()
        print("  ✅ RAG pipeline created successfully")
        
        # Test system stats
        stats = rag.get_system_stats()
        print(f"  ✅ System stats retrieved: {stats['model_name']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RAG pipeline test failed: {e}")
        return False

def test_data_directory():
    """Test that data directory exists and check for PDF files."""
    print("\n🧪 Testing data directory...")
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("  ⚠️  Data directory does not exist - creating it")
        data_dir.mkdir()
        print("  ✅ Data directory created")
    else:
        print("  ✅ Data directory exists")
    
    # Check for PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        print(f"  ✅ Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files[:3]:  # Show first 3
            print(f"     - {pdf_file.name}")
        if len(pdf_files) > 3:
            print(f"     ... and {len(pdf_files) - 3} more")
    else:
        print("  ⚠️  No PDF files found in data directory")
        print("     Add PDF files to test document processing")
    
    return True

def main():
    """Run all tests."""
    print("Financial RAG System - Test Suite")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Directory", test_data_directory),
        ("Document Processor", test_document_processor),
        ("Embedding Manager", test_embedding_manager),
        ("RAG Pipeline", test_rag_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to use.")
        return True
    else:
        print("😞 Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ You can now run: streamlit run app.py")
        else:
            print("\n❌ Please fix the issues above before running the application")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
    
    input("\nPress Enter to continue...") 