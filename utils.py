import cv2
import numpy as np
from PIL import Image
import io


def validate_file(uploaded_file):
    """
    Validate uploaded file for type and size constraints
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    # Check file type
    allowed_types = ['png', 'jpg', 'jpeg', 'pdf']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_types:
        return False, f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_types)}"
    
    # Check file size (limit to 50MB)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    if uploaded_file.size > max_size:
        return False, f"File too large: {uploaded_file.size / (1024*1024):.1f}MB. Maximum allowed: 50MB"
    
    # Check if file is not empty
    if uploaded_file.size == 0:
        return False, "File is empty"
    
    return True, ""


def preprocess_image(image):
    """
    Preprocess image for better OCR accuracy
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    try:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply different preprocessing techniques and combine results
        
        # Method 1: Simple thresholding
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Method 4: Noise removal
        denoised = cv2.medianBlur(cleaned, 3)
        
        # Choose the best preprocessing result
        # For now, return the denoised result
        return denoised
        
    except Exception as e:
        # If preprocessing fails, return original grayscale
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def enhance_image_contrast(image):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Grayscale image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    try:
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    except Exception:
        return image


def resize_image_for_ocr(image, max_width=2000, max_height=2000):
    """
    Resize image to optimal size for OCR while maintaining aspect ratio
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        numpy.ndarray: Resized image
    """
    try:
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
        
    except Exception:
        return image


def preprocess_for_telugu_ocr(image):
    """
    Specialized preprocessing for Telugu text recognition
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Preprocessed image optimized for Telugu OCR
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small or too large
        gray = resize_image_for_ocr(gray)
        
        # Enhance contrast
        enhanced = enhance_image_contrast(gray)
        
        # Apply Gaussian blur to smooth text
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
        
    except Exception:
        # Fallback to basic preprocessing
        return preprocess_image(image)
