#!/usr/bin/env python3
"""
Model downloader script for Financial RAG System.
Pre-downloads models to enable offline usage.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for model downloading."""
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    logger.info("Environment variables set for model downloading")

def download_sentence_transformer_models():
    """Download SentenceTransformer models."""
    models_to_try = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2", 
        "msmarco-MiniLM-L6-cos-v5",
        "multi-qa-MiniLM-L6-cos-v1"
    ]
    
    successful_downloads = []
    
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to download: {model_name}")
            
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            
            logger.info(f"✅ Successfully downloaded: {model_name}")
            successful_downloads.append(model_name)
            
            # Test the model
            test_embedding = model.encode(["This is a test sentence."])
            logger.info(f"✅ Model {model_name} is working correctly")
            
        except Exception as e:
            logger.warning(f"❌ Failed to download {model_name}: {e}")
            continue
    
    return successful_downloads

def download_language_models():
    """Download language models."""
    models_to_try = [
        "google/flan-t5-base",
        "google/flan-t5-small"
    ]
    
    successful_downloads = []
    
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to download: {model_name}")
            
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✅ Downloaded tokenizer for: {model_name}")
            
            # Download model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info(f"✅ Downloaded model: {model_name}")
            
            successful_downloads.append(model_name)
            
        except Exception as e:
            logger.warning(f"❌ Failed to download {model_name}: {e}")
            continue
    
    return successful_downloads

def main():
    """Main function to download all models."""
    print("Financial RAG System - Model Downloader")
    print("=" * 50)
    print()
    
    # Setup environment
    setup_environment()
    
    # Check if we have internet connectivity
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code != 200:
            raise Exception("Cannot reach Hugging Face")
        logger.info("✅ Internet connectivity confirmed")
    except Exception as e:
        logger.error(f"❌ No internet connectivity: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    print("Downloading models... This may take several minutes.")
    print()
    
    # Download SentenceTransformer models
    print("📥 Downloading embedding models...")
    sentence_models = download_sentence_transformer_models()
    
    print()
    print("📥 Downloading language models...")
    language_models = download_language_models()
    
    print()
    print("=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    
    if sentence_models:
        print("✅ Successfully downloaded embedding models:")
        for model in sentence_models:
            print(f"   - {model}")
    else:
        print("❌ No embedding models were downloaded successfully")
    
    print()
    
    if language_models:
        print("✅ Successfully downloaded language models:")
        for model in language_models:
            print(f"   - {model}")
    else:
        print("❌ No language models were downloaded successfully")
    
    print()
    
    if sentence_models or language_models:
        print("🎉 Model download completed!")
        print("You can now run the application offline using the downloaded models.")
        return True
    else:
        print("😞 No models were downloaded successfully.")
        print("The application will use TF-IDF fallback for embeddings.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nTo run the application: streamlit run app.py")
        else:
            print("\nTrying to run without pre-downloaded models...")
            print("The application will attempt to download models on first use.")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nAn error occurred: {e}")
    
    input("\nPress Enter to continue...") 