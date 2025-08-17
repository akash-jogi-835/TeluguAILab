import streamlit as st
import os
import tempfile
from ocr_processor import OCRProcessor
from nlp_processor import NLPProcessor
from utils import validate_file, preprocess_image

# -----------------------------------------------------------
# Initialize processors with optional LLM
# -----------------------------------------------------------
@st.cache_resource
def load_processors(provider: str, model_name: str, temperature: float, openai_key: str | None):
    """Load OCR and NLP processors with optional LLM client"""
    ocr_processor = OCRProcessor()

    llm_client = None
    if provider in ("openai", "local"):
        from llm_client import LLMClient
        llm_client = LLMClient(
            provider=provider,
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_key or None,
        )

    nlp_processor = NLPProcessor(llm_client=llm_client)
    return ocr_processor, nlp_processor

# -----------------------------------------------------------
# Main App
# -----------------------------------------------------------
def main():
    st.title("Telugu OCR + NLP (with LLM support)")
    st.markdown("Upload an image or PDF to extract Telugu/English text, summarize it, or ask questions.")

    # Sidebar config
    st.sidebar.header("Configuration")

    # OCR options
    ocr_languages = st.sidebar.selectbox(
        "OCR Languages",
        ["tel+eng", "tel", "eng"],
        index=0,
        help="Select OCR language combination"
    )
    ocr_mode = st.sidebar.selectbox(
        "OCR Mode",
        ["Auto (Multiple attempts)", "Standard", "Single block", "Single word"],
        index=0,
        help="Select OCR mode"
    )

    # NLP / LLM options
    st.sidebar.subheader("NLP / LLM")
    nlp_mode = st.sidebar.selectbox(
        "Processing Backend",
        ["basic", "advanced (LLM)"],
        index=0,
        help="Choose offline (basic) or LLM-backed processing"
    )

    provider = st.sidebar.selectbox("LLM Provider", ["openai", "local"], index=0)
    model_name = st.sidebar.text_input(
        "LLM Model",
        value="gpt-4o-mini" if provider == "openai" else "google/flan-t5-base",
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    openai_key = st.sidebar.text_input("OPENAI_API_KEY", type="password") if provider == "openai" else ""

    # Load processors
    try:
        ocr_processor, nlp_processor = load_processors(provider, model_name, temperature, openai_key)
    except Exception as e:
        st.error(f"Failed to initialize processors: {str(e)}")
        st.stop()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Supported: PNG, JPG, JPEG, PDF"
    )

    if uploaded_file is not None:
        # Validate file
        is_valid, error_message = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_message)
            return

        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Process file
        with st.spinner("Running OCR..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # OCR extraction
                extracted_text = ocr_processor.extract_text(tmp_file_path, languages=ocr_languages, mode=ocr_mode)

                os.unlink(tmp_file_path)  # cleanup temp file

                if not extracted_text.strip():
                    st.warning("No text extracted. Try another file or adjust OCR settings.")
                    return

                # Show extracted text
                st.header("Extracted Text")
                st.text_area("Raw Text", extracted_text, height=200, disabled=True)

                # NLP Section
                st.header("NLP Processing")
                tab1, tab2 = st.tabs(["Summarization", "Q&A"])

                with tab1:
                    st.subheader("Summarization")
                    if st.button("Generate Summary", key="summarize"):
                        with st.spinner("Summarizing..."):
                            try:
                                summary = nlp_processor.summarize_text(
                                    extracted_text,
                                    model_name=nlp_mode
                                )
                                st.success("Done!")
                                st.text_area("Summary", summary, height=150, disabled=True)
                            except Exception as e:
                                st.error(f"Summarization failed: {str(e)}")

                with tab2:
                    st.subheader("Ask a Question")
                    question = st.text_input("Your question about the text:")
                    if question and st.button("Get Answer", key="qa"):
                        with st.spinner("Generating answer..."):
                            try:
                                answer = nlp_processor.answer_question(
                                    extracted_text,
                                    question,
                                    model_name=nlp_mode
                                )
                                st.success("Done!")
                                st.text_area("Answer", answer, height=120, disabled=True)
                            except Exception as e:
                                st.error(f"Q&A failed: {str(e)}")

                # Download & copy options
                st.header("Export Options")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Copy Extracted Text"):
                        st.code(extracted_text, language=None)
                        st.info("Text displayed above for easy copying")

                with col2:
                    st.download_button(
                        label="Download as .txt",
                        data=extracted_text,
                        file_name=f"extracted_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                try:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except:
                    pass

    else:
        st.info("Upload an image or PDF to begin.")

if __name__ == "__main__":
    main()
