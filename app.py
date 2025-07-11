"""
Main Streamlit application for Financial Q&A RAG System.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
import time
import os
from pathlib import Path

# Import our modules
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.utils import (
    get_pdf_files, 
    format_file_size, 
    get_file_info, 
    format_sources,
    ensure_directory_exists
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Q&A Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stAlert {
        margin-top: 1rem;
    }
    
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .confidence-score {
        color: #28a745;
        font-weight: bold;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with caching."""
    try:
        with st.spinner("Initializing RAG pipeline... This may take a few minutes for the first run."):
            # Ensure directories exist
            ensure_directory_exists("data")
            ensure_directory_exists("vector_store")
            
            # Create RAG pipeline
            rag = create_rag_pipeline(
                model_name="google/flan-t5-base",
                embedding_model="all-MiniLM-L6-v2",
                vector_store_path="vector_store"
            )
            
            logger.info("RAG pipeline initialized successfully")
            return rag
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def process_documents(rag_pipeline: RAGPipeline, data_directory: str = "data") -> bool:
    """Process PDF documents and add them to the vector store."""
    try:
        # Get PDF files
        pdf_files = get_pdf_files(data_directory)
        
        if not pdf_files:
            st.warning(f"No PDF files found in the '{data_directory}' directory.")
            return False
        
        # Display files to be processed
        st.info(f"Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            file_info = get_file_info(pdf_file)
            st.write(f"- {file_info['name']} ({file_info['size_formatted']})")
        
        # Process documents
        with st.spinner("Processing documents... This may take several minutes."):
            progress_bar = st.progress(0)
            
            success = rag_pipeline.process_documents(data_directory)
            
            progress_bar.progress(100)
            
            if success:
                st.success("✅ Documents processed successfully!")
                return True
            else:
                st.error("❌ Failed to process documents.")
                return False
                
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        st.error(f"Error processing documents: {e}")
        return False

def display_answer(result: Dict[str, Any]):
    """Display the answer with sources in a formatted way."""
    
    # Display the answer
    st.markdown("### 🤖 Answer")
    st.markdown(result['answer'])
    
    # Display confidence score
    if result['confidence'] > 0:
        confidence_percentage = result['confidence'] * 100
        st.markdown(f"**Confidence Score:** <span class='confidence-score'>{confidence_percentage:.1f}%</span>", 
                   unsafe_allow_html=True)
    
    # Display sources
    if result['sources']:
        st.markdown("### 📚 Sources")
        
        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"Source {i}: {source['source_file']} (Pages: {source['page_range']})"):
                st.markdown(f"**File:** {source['source_file']}")
                st.markdown(f"**Pages:** {source['page_range']}")
                st.markdown(f"**Similarity Score:** {source['similarity_score']:.3f}")
                st.markdown("**Text Preview:**")
                st.text(source['text_preview'])
    else:
        st.info("No sources available for this answer.")

def main():
    """Main application function."""
    
    # Header
    st.markdown("<h1 class='main-header'>💰 Financial Q&A Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Ask questions about your financial documents using AI-powered retrieval and analysis.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 System Controls")
        
        # Initialize RAG pipeline
        if st.button("🚀 Initialize System", type="primary"):
            st.session_state.rag_pipeline = initialize_rag_pipeline()
            if st.session_state.rag_pipeline:
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system.")
        
        st.markdown("---")
        
        # Document management
        st.markdown("## 📁 Document Management")
        
        # Check if data directory exists and show files
        if os.path.exists("data"):
            pdf_files = get_pdf_files("data")
            if pdf_files:
                st.success(f"Found {len(pdf_files)} PDF files in data directory")
                for pdf_file in pdf_files:
                    file_info = get_file_info(pdf_file)
                    st.write(f"• {file_info['name']} ({file_info['size_formatted']})")
            else:
                st.info("No PDF files found in data directory")
        else:
            st.warning("Data directory not found. Please create a 'data' folder and add PDF files.")
        
        # Process documents button
        if st.button("📊 Process Documents"):
            if st.session_state.rag_pipeline:
                success = process_documents(st.session_state.rag_pipeline)
                if success:
                    st.session_state.documents_processed = True
            else:
                st.error("Please initialize the system first.")
        
        # Clear documents button
        if st.button("🗑️ Clear Documents"):
            if st.session_state.rag_pipeline:
                with st.spinner("Clearing documents..."):
                    success = st.session_state.rag_pipeline.clear_documents()
                    if success:
                        st.success("Documents cleared successfully!")
                        st.session_state.documents_processed = False
                    else:
                        st.error("Failed to clear documents.")
            else:
                st.error("Please initialize the system first.")
        
        st.markdown("---")
        
        # System status
        st.markdown("## 📊 System Status")
        
        if st.session_state.rag_pipeline:
            try:
                stats = st.session_state.rag_pipeline.get_system_stats()
                st.write(f"**Model:** {stats['model_name']}")
                st.write(f"**Device:** {stats['device']}")
                st.write(f"**Documents:** {stats['embedding_stats']['total_chunks']} chunks")
                st.write(f"**Source Files:** {stats['embedding_stats']['unique_source_files']}")
            except Exception as e:
                st.error(f"Error getting stats: {e}")
        else:
            st.info("System not initialized")
        
        st.markdown("---")
        
        # Settings
        st.markdown("## ⚙️ Settings")
        
        n_retrieve = st.slider("Number of sources to retrieve", 1, 10, 5)
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_sources = st.checkbox("Show source details", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check system status
        if not st.session_state.rag_pipeline:
            st.warning("⚠️ Please initialize the system using the sidebar before asking questions.")
            st.info("👈 Click 'Initialize System' in the sidebar to get started.")
            return
        
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please process documents before asking questions.")
            st.info("👈 Click 'Process Documents' in the sidebar to load your PDF files.")
            return
        
        # Question input
        st.markdown("## 💬 Ask a Question")
        
        # Text input for question
        user_question = st.text_input(
            "Enter your question about the financial documents:",
            placeholder="e.g., What was the revenue growth last quarter?",
            key="user_question"
        )
        
        # Submit button
        col_submit, col_clear = st.columns([1, 1])
        
        with col_submit:
            if st.button("🔍 Get Answer", type="primary"):
                if user_question.strip():
                    # Process the question
                    with st.spinner("Searching documents and generating answer..."):
                        try:
                            result = st.session_state.rag_pipeline.answer_question(
                                user_question, 
                                n_retrieve=n_retrieve
                            )
                            
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                'question': user_question,
                                'answer': result['answer'],
                                'sources': result['sources'],
                                'confidence': result['confidence'],
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Display the answer
                            display_answer(result)
                            
                        except Exception as e:
                            st.error(f"Error processing question: {e}")
                            logger.error(f"Error processing question: {e}")
                else:
                    st.warning("Please enter a question.")
        
        with col_clear:
            if st.button("🧹 Clear History"):
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
    
    with col2:
        # Quick actions
        st.markdown("## 🚀 Quick Actions")
        
        example_questions = [
            "What is the current revenue?",
            "How much debt does the company have?",
            "What are the main expenses?",
            "What is the profit margin?",
            "How did performance compare to last year?"
        ]
        
        st.markdown("**Example Questions:**")
        for question in example_questions:
            if st.button(f"💡 {question}", key=f"example_{question}"):
                st.session_state.user_question = question
                st.rerun()
    
    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("## 📝 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:50]}..."):
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Answer:** {conv['answer']}")
                st.markdown(f"**Time:** {conv['timestamp']}")
                if show_confidence and conv['confidence'] > 0:
                    st.markdown(f"**Confidence:** {conv['confidence']*100:.1f}%")
                if show_sources and conv['sources']:
                    st.markdown(f"**Sources:** {len(conv['sources'])} documents")
    
    # Footer
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. 🚀 Initialize the system using the sidebar")
    st.markdown("2. 📁 Add PDF files to the 'data' directory")
    st.markdown("3. 📊 Process documents to create the knowledge base")
    st.markdown("4. 💬 Ask questions about your financial documents")
    
    # System info
    with st.expander("🔍 System Information"):
        if st.session_state.rag_pipeline:
            try:
                stats = st.session_state.rag_pipeline.get_system_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting system info: {e}")
        else:
            st.info("System not initialized")

if __name__ == "__main__":
    main() 