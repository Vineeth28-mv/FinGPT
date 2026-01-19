# Financial Q&A RAG System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that functions as a "Financial Analyst Assistant" to answer questions based on PDF financial documents. The system provides accurate, verifiable answers grounded in the provided texts with proper source citations.

## 🎯 Project Overview

This RAG system combines the power of document retrieval with language model generation to create an intelligent financial assistant. It processes PDF financial documents, creates searchable embeddings, and uses a language model to generate contextually accurate answers with proper source citations.

### Key Features

-  **PDF Document Processing**: Robust text extraction from financial PDFs
-  **Intelligent Retrieval**: Semantic search using sentence transformers
-  **AI-Powered Q&A**: Context-aware answer generation with FLAN-T5
-  **Source Citations**: Proper attribution with page numbers and document references
-  **Web Interface**: Clean, intuitive Streamlit-based user interface
-  **Persistent Storage**: ChromaDB vector database for efficient retrieval


## 📋 Design Choices

### 1. PDF Processing - PDFplumber

**Choice**: `pdfplumber` for PDF text extraction

**Rationale**:
- **Superior Layout Handling**: Better at handling multi-column layouts common in financial documents
- **Table Extraction**: Excellent support for extracting tabular data from financial reports
- **Metadata Preservation**: Maintains spatial information about text positioning
- **Robust Error Handling**: More resilient to corrupted or complex PDF structures

**Alternative Considered**: PyPDF2 - rejected due to poor handling of complex layouts

### 2. Chunking Strategy

**Choice**: 1000-word chunks with 200-word overlap

**Rationale**:
- **Context Preservation**: 1000 words provide sufficient context for financial concepts
- **Overlap Benefits**: 200-word overlap ensures important information isn't lost at chunk boundaries
- **Performance Balance**: Optimal balance between context richness and processing speed
- **Sentence Boundaries**: Algorithm attempts to break at sentence boundaries to maintain coherence

**Alternative Considered**: Fixed-size character chunks - rejected due to loss of semantic coherence

### 3. Embedding Model - all-MiniLM-L6-v2

**Choice**: SentenceTransformer `all-MiniLM-L6-v2`

**Rationale**:
- **Free & Open Source**: No API costs or usage limits
- **Efficient Performance**: Good balance of quality and speed (22MB model)
- **Financial Domain**: Performs well on business and financial text
- **Multilingual Support**: Handles various languages if needed
- **Active Community**: Well-maintained with regular updates

**Alternative Considered**: OpenAI embeddings - rejected due to API costs and privacy concerns

### 4. Language Model - FLAN-T5

**Choice**: Google's `FLAN-T5-base` or `FLAN-T5-large`

**Rationale**:
- **Instruction Following**: Excellent at following specific prompts and instructions
- **Free & Open Source**: No API costs or rate limits
- **Reasoning Capabilities**: Good at logical reasoning required for financial analysis
- **Factual Accuracy**: Trained to provide accurate, grounded responses
- **Size Options**: Base (250M) and Large (780M) variants for different resource constraints

**Alternative Considered**: GPT-3.5/4 - rejected due to API costs and data privacy concerns

### 5. Vector Store - ChromaDB

**Choice**: ChromaDB for vector storage and retrieval

**Rationale**:
- **Simplicity**: Minimal setup required, works out of the box
- **Persistence**: Automatic persistence without additional configuration
- **Performance**: Efficient similarity search with HNSW indexing
- **Metadata Support**: Rich metadata filtering capabilities
- **Local Storage**: Data remains on user's machine for privacy
- **Active Development**: Regular updates and improvements

**Alternative Considered**: Pinecone - rejected due to cloud dependency and costs

### 6. User Interface - Streamlit

**Choice**: Streamlit for web interface

**Rationale**:
- **Rapid Development**: Quick to build and iterate on UI components
- **Python Native**: Seamless integration with Python ML/AI stack
- **Interactive Components**: Rich widgets for user interaction
- **Deployment Ready**: Easy deployment options for sharing
- **Community**: Large community with extensive documentation

**Alternative Considered**: Gradio - rejected due to less flexibility for complex layouts

## 🚀 Installation Instructions

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- 2GB+ free disk space

### 🎯 **Easy Setup (Recommended)**

#### **For Windows:**
1. **Run setup**: Double-click `setup.bat` or run:
   ```cmd
   setup.bat
   ```

2. **Download models** (optional, for better performance):
   ```cmd
   download_models.bat
   ```

3. **Add PDF files** to the `data/` directory

4. **Run the application**: Double-click `run_app.bat` or run:
   ```cmd
   run_app.bat
   ```

#### **For macOS/Linux:**
1. **Run setup**:
   ```bash
   ./setup.sh
   ```

2. **Download models** (optional, for better performance):
   ```bash
   python download_models.py
   ```

3. **Add PDF files** to the `data/` directory

4. **Run the application**:
   ```bash
   ./run_app.sh
   ```

### 🔧 **Manual Setup**

If you prefer manual setup or encounter issues with the automated scripts:

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd financial_rag_system
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Set Environment Variables (Important!)

```bash
# On Windows (in Command Prompt):
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
set PYTHONHTTPSVERIFY=0

# On macOS/Linux (in Terminal):
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0
```

#### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt --upgrade
```

#### Step 5: Set Up Directories

```bash
# Create data directory for PDF files
mkdir data

# Vector store directory will be created automatically
```

#### Step 6: Add Financial Documents

Place your PDF financial documents in the `data/` directory:

```
data/
├── financial_report_2023.pdf
├── quarterly_earnings_q4.pdf
├── annual_report_2022.pdf
└── balance_sheet_2023.pdf
```

## 📖 Usage Instructions

### 🎯 **Easy Way (Recommended)**

**Windows:** Double-click `run_app.bat`

**macOS/Linux:** Run `./run_app.sh`

### 🔧 **Manual Way**

1. **Set environment variables** (important for SSL issues):
   ```bash
   # Windows:
   set HF_HUB_DISABLE_SYMLINKS_WARNING=1
   set PYTHONHTTPSVERIFY=0
   
   # macOS/Linux:
   export HF_HUB_DISABLE_SYMLINKS_WARNING=1
   export PYTHONHTTPSVERIFY=0
   ```

2. **Activate virtual environment**:
   ```bash
   # Windows:
   venv\Scripts\activate
   
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```


## 🎛️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LANGUAGE_MODEL=google/flan-t5-base
VECTOR_STORE_PATH=vector_store

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONTEXT_LENGTH=1000

# Application Configuration
LOG_LEVEL=INFO
```

### Advanced Configuration

Modify `src/utils.py` for custom settings:

```python
# Chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

# Model parameters
DEFAULT_MODEL_NAME = "google/flan-t5-base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### SSL Certificate Issues

If you encounter SSL certificate verification errors:

1. **Use the automated scripts** (recommended):
   - Run `setup.bat` / `setup.sh` which handles SSL issues automatically
   - Run `run_app.bat` / `run_app.sh` which sets proper environment variables

2. **Manual SSL fix**:
   ```bash
   # Set these environment variables before running
   set PYTHONHTTPSVERIFY=0
   set CURL_CA_BUNDLE=
   set REQUESTS_CA_BUNDLE=
   ```

3. **Pre-download models**:
   ```bash
   # Download models when you have better network access
   download_models.bat  # Windows
   python download_models.py  # Linux/Mac
   ```

4. **Fallback mode**: The system will automatically use TF-IDF embeddings if all model downloads fail

### Model Download Issues

If model downloads fail:

1. **Check internet connection**
2. **Try the model downloader**: `download_models.bat` or `python download_models.py`
3. **Use different network**: Try from a different location/network
4. **Fallback embeddings**: The system will work with TF-IDF if needed

### Memory Issues

If you run out of memory:

1. **Reduce chunk size** in `src/utils.py`: `DEFAULT_CHUNK_SIZE = 500`
2. **Use smaller model**: Change to `google/flan-t5-small` in the code
3. **Process fewer documents** at once

### Performance Issues

If the system is slow:

1. **Pre-download models** using `download_models.bat`
2. **Use GPU** if available (CUDA-enabled PyTorch)
3. **Reduce number of retrieved chunks** in the UI settings


1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m "Add new feature"`
5. **Push to the branch**: `git push origin feature/new-feature`
6. **Create a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



