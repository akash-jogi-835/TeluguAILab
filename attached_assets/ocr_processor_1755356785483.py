import pytesseract
import cv2
import numpy as np
from PIL import Image
import pdf2image
import os
import tempfile
from utils import preprocess_image
import cv2



class OCRProcessor:
    """Handles OCR text extraction from images and PDFs"""
    
    def __init__(self):
        """Initialize OCR processor with Telugu language support"""
        # Set Tesseract configuration for better Telugu recognition
        self.config = r'--oem 3 --psm 6'
        
    def extract_text_from_image(self, image_path, languages='tel+eng', mode='Auto (Multiple attempts)'):
        """
        Extract text from a single image using Tesseract OCR
        
        Args:
            image_path (str): Path to the image file
            languages (str): Language codes for OCR (e.g., 'tel+eng')
            mode (str): OCR processing mode
            
        Returns:
            str: Extracted text
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image file")
            
            # Get PSM configurations based on mode
            psm_configs = self._get_psm_configs(mode)
            
            # Try multiple preprocessing approaches
            results = []
            
            for config in psm_configs:
                # Method 1: Basic preprocessing
                processed_image1 = preprocess_image(image)
                pil_image1 = Image.fromarray(processed_image1)
                text1 = pytesseract.image_to_string(pil_image1, lang=languages, config=config)
                results.append(text1.strip())
                
                # Method 2: Try with original image
                pil_image2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                text2 = pytesseract.image_to_string(pil_image2, lang=languages, config=config)
                results.append(text2.strip())
            
            # Choose the best result (longest meaningful text with fewer artifacts)
            best_text = self._select_best_result(results)
            
            return best_text if best_text else "No text could be extracted from the image."
            
        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")
    
    def _get_psm_configs(self, mode):
        """Get PSM configurations based on user-selected mode"""
        if mode == "Auto (Multiple attempts)":
            return [
                r'--oem 3 --psm 6',  # Uniform block of text
                r'--oem 3 --psm 8',  # Single word
                r'--oem 3 --psm 7',  # Single text line
                r'--oem 3 --psm 4',  # Single column of text
                r'--oem 3 --psm 3'   # Fully automatic page segmentation
            ]
        elif mode == "Standard":
            return [r'--oem 3 --psm 6']
        elif mode == "Single block":
            return [r'--oem 3 --psm 6']
        elif mode == "Single word":
            return [r'--oem 3 --psm 8']
        else:
            return [r'--oem 3 --psm 6']
    
    def _select_best_result(self, results):
        """Select the best OCR result from multiple attempts"""
        if not results:
            return ""
        
        # Filter out results that are mostly garbage
        filtered_results = []
        for text in results:
            clean_text = text.replace(' ', '').replace('\n', '').replace('\t', '')
            
            # Check if text has reasonable content
            if len(clean_text) > 2:
                # Count alphanumeric vs special characters
                alpha_count = sum(1 for c in clean_text if c.isalnum())
                total_count = len(clean_text)
                
                # If at least 30% of characters are alphanumeric, consider it valid
                if total_count > 0 and (alpha_count / total_count) >= 0.3:
                    filtered_results.append(text)
        
        if not filtered_results:
            # If no good results, return the longest original result
            return max(results, key=len) if results else ""
        
        # Return the longest filtered result
        return max(filtered_results, key=len)
    
    def extract_text_from_pdf(self, pdf_path, languages='tel+eng', mode='Auto (Multiple attempts)'):
        """
        Extract text from PDF by converting to images first
        
        Args:
            pdf_path (str): Path to the PDF file
            languages (str): Language codes for OCR
            mode (str): OCR processing mode
            
        Returns:
            str: Extracted text from all pages
        """
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            extracted_texts = []
            
            for i, image in enumerate(images):
                # Save image temporarily
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name, 'PNG')
                    
                    # Extract text from image
                    text = self.extract_text_from_image(tmp_file.name, languages, mode)
                    extracted_texts.append(text)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            # Combine text from all pages
            combined_text = '\n\n--- Page Break ---\n\n'.join(extracted_texts)
            return combined_text.strip()
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def extract_text(self, file_path, languages='tel+eng', mode='Auto (Multiple attempts)'):
        """
        Extract text from file (image or PDF)
        
        Args:
            file_path (str): Path to the file
            languages (str): Language codes for OCR
            mode (str): OCR processing mode
            
        Returns:
            str: Extracted text
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path, languages, mode)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            return self.extract_text_from_image(file_path, languages, mode)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def get_text_confidence(self, image_path, languages='tel+eng'):
        """
        Get OCR confidence scores for debugging
        
        Args:
            image_path (str): Path to the image file
            languages (str): Language codes for OCR
            
        Returns:
            dict: OCR data with confidence scores
        """
        try:
            image = cv2.imread(image_path)
            processed_image = preprocess_image(image)
            pil_image = Image.fromarray(processed_image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                pil_image,
                lang=languages,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            return data
            
        except Exception as e:
            raise Exception(f"Confidence analysis failed: {str(e)}")
