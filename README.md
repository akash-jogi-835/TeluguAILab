# Telugu OCR + NLP Processing Application

## Overview

This is a Streamlit web application that performs Optical Character Recognition (OCR) on Telugu and English text from images and PDFs, followed by Natural Language Processing operations like summarization and question-answering. The system supports both local processing and cloud-based LLM integration for advanced text analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **User Interface**: Single-page application with sidebar configuration
- **File Handling**: Supports image uploads (PNG, JPG, JPEG) and PDF files up to 50MB
- **Caching**: Uses Streamlit's `@st.cache_resource` for efficient processor initialization

### OCR Processing Layer
- **Engine**: Tesseract OCR with pytesseract Python wrapper
- **Language Support**: Telugu (`tel`) and English (`eng`) with configurable combinations
- **Processing Modes**: Multiple OCR strategies including auto-detection, standard, single block, and single word modes
- **Image Preprocessing**: OpenCV-based enhancement for better text recognition accuracy
- **PDF Support**: Converts PDF pages to images using pdf2image before OCR processing

### NLP Processing Layer
- **Dual Architecture**: Supports both local and cloud-based processing
- **Local Processing**: Fallback extractive summarization using word frequency analysis
- **Cloud Integration**: Optional LLM client supporting OpenAI GPT models and local transformers
- **Multilingual Support**: Handles both Telugu and English text with language-specific stop words
- **Features**: Text summarization, question-answering, and basic text analysis

### LLM Integration
- **Primary Provider**: OpenAI API with GPT-4o-mini as default model
- **Fallback Option**: Local transformers pipeline for offline processing
- **Configuration**: Flexible provider selection with temperature and token limit controls
- **Error Handling**: Graceful degradation when LLM services are unavailable

### Utility Layer
- **File Validation**: Comprehensive checks for file type, size, and integrity
- **Image Processing**: OpenCV-based preprocessing including noise reduction, contrast enhancement, and scaling
- **Error Handling**: Robust validation and preprocessing pipelines

## External Dependencies

### Core Processing Libraries
- **pytesseract**: Tesseract OCR Python wrapper for text extraction
- **opencv-python**: Computer vision library for image preprocessing
- **pdf2image**: PDF to image conversion for OCR processing
- **PIL (Pillow)**: Python Imaging Library for image manipulation

### Machine Learning & NLP
- **transformers**: Hugging Face transformers library for local NLP models
- **openai**: Official OpenAI Python client for GPT model access

### Web Framework
- **streamlit**: Web application framework for the user interface
- **tempfile**: Python standard library for temporary file handling

### Optional Services
- **OpenAI API**: Cloud-based language model service requiring API key
- **CUDA**: Optional GPU acceleration for local transformer models