# Telugu OCR Text Extraction & NLP Processing

## Overview

This is a Streamlit-based web application that performs Optical Character Recognition (OCR) on Telugu text from images and PDFs, followed by Natural Language Processing operations. The system is designed to handle multilingual content with a focus on Telugu language support, combining computer vision and NLP capabilities to extract and process text content.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **OCR Layer**: Tesseract-based text extraction with Telugu language support
- **NLP Layer**: Transformer-based models for text processing and analysis
- **Utilities**: Helper functions for file validation and image preprocessing

The architecture is designed as a single-node application with in-memory processing, suitable for demonstration and small-scale usage.

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Entry point and user interface controller
- **Technology**: Streamlit framework
- **Features**: File upload, configuration options, processor initialization
- **Caching**: Uses Streamlit's `@st.cache_resource` for processor loading

### 2. OCR Processor (ocr_processor.py)
- **Purpose**: Text extraction from images and PDFs
- **Technology**: Tesseract OCR with pytesseract wrapper
- **Language Support**: Telugu (`tel`) and English (`eng`) with configurable combinations
- **Input Formats**: PNG, JPG, JPEG, PDF files

### 3. NLP Processor (nlp_processor.py)
- **Purpose**: Multilingual text processing and analysis
- **Technology**: Hugging Face Transformers
- **Models**: 
  - MT5 (Multilingual T5) for conditional generation tasks
  - AutoModel/AutoTokenizer for general NLP tasks
- **GPU Support**: CUDA-enabled with CPU fallback

### 4. Utilities (utils.py)
- **Purpose**: Common helper functions
- **Functions**:
  - File validation (size and type checking)
  - Image preprocessing for OCR optimization
  - OpenCV-based image enhancement

## Data Flow

1. **File Upload**: User uploads image/PDF through Streamlit interface
2. **Validation**: File type and size validation using utility functions
3. **Preprocessing**: Image enhancement for better OCR accuracy
4. **OCR Processing**: Text extraction using Tesseract with Telugu language models
5. **NLP Processing**: Text analysis using transformer-based models
6. **Results Display**: Processed results shown in the web interface

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **pytesseract**: Python wrapper for Tesseract OCR
- **transformers**: Hugging Face transformer models
- **torch**: PyTorch for deep learning models
- **opencv-python**: Image processing and computer vision
- **PIL (Pillow)**: Image manipulation
- **pdf2image**: PDF to image conversion

### Language Models
- **Tesseract**: For OCR with Telugu language support
- **MT5**: Multilingual T5 model for text generation
- **AutoModel**: Various Hugging Face models for NLP tasks

### System Requirements
- **Tesseract OCR**: Must be installed on the system
- **Telugu Language Data**: Tesseract Telugu language pack
- **CUDA (Optional)**: For GPU acceleration of NLP models

## Deployment Strategy

The application is designed for local deployment or cloud hosting with the following considerations:

### Local Development
- Python environment with all dependencies installed
- Tesseract OCR with Telugu language support
- Streamlit development server

### Production Deployment
- **Containerization**: Suitable for Docker deployment
- **Memory Requirements**: Sufficient RAM for loading transformer models
- **Storage**: Temporary file storage for uploaded files
- **GPU Support**: Optional CUDA support for faster NLP processing

### Configuration Options
- OCR language selection (Telugu, English, or combined)
- NLP model selection for different processing tasks
- File size limits and supported formats
- Processing parameters for image enhancement

The modular design allows for easy scaling and modification of individual components without affecting the overall system architecture.